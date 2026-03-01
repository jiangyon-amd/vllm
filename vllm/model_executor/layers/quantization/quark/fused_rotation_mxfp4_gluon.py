#!/usr/bin/env python3
"""
Fused Rotation + MXFP4 Quant v13 — No convert_layout, HIP-compatible

Key optimizations:
  1. Quantization directly in MFMA layout (no convert_layout before quant)
  2. Hardware v_cvt_scalef32_pk_fp4_f32 instruction for FP4 conversion
Numerically aligned with HIP per_1x32_f4_quant_hip:
  - Scale: round-up (ceiling) to power-of-2, matching HIP carry logic
  - FP4 conversion: gfx950 hardware instruction, bit-exact with HIP vec_convert<fp4>

MFMA layout for [32, 128] with warps=[2,2], tiles=[1,4], transposed=True:
  - Each thread holds 4 cols per tile × 4 tiles = 16 elements per warp
  - Quant group (32 cols) spans 2 tiles → each thread holds 8 of 32 values
  - 4 lane-groups per warp cooperate on same row
  - abs_max: thread-local max of 8 values + 4-way warp shuffle → NO LDS!

This eliminates the expensive convert_layout(MFMA→BlockedLayout) which was
~30% of the kernel time.
"""

import torch
import triton
from triton import language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import time


@gluon.jit
def _fused_rot_quant_v13(
    x_ptr, rot_ptr, fp4_ptr, scale_ptr,
    M, K,
    stride_x_m, stride_rot_r, stride_rot_c,
    stride_fp4_m, stride_sc_m,
    RS: gl.constexpr,
    QG: gl.constexpr,
    BLOCK_M: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    pid_m   = gl.program_id(0)
    pid_rot = gl.program_id(1)

    # ======== Layouts ========
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16], transposed=True,
        warps_per_cta=[2, 2], tiles_per_warp=[1, 4],
    )
    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4], threads_per_warp=[16, 4],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0],
    )
    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[4, 1], threads_per_warp=[4, 16],
        warps_per_cta=[1, NUM_WARPS], order=[0, 1],
    )
    shared_rot: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[1, 0],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=4)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=4)

    NUM_QG: gl.constexpr = RS // QG
    HALF_QG: gl.constexpr = QG // 2

    # ======== Offsets ========
    m_base = pid_m * BLOCK_M
    col_base = pid_rot * RS
    offs_m = m_base + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked_mk))
    m_mask = offs_m < M

    # ======== Rotation matmul (single-shot, no K-loop) ========
    offs_k = gl.arange(0, RS, layout=gl.SliceLayout(0, blocked_mk))
    x_offs = offs_m[:, None] * stride_x_m + (col_base + offs_k)[None, :]
    x_tile = gl.amd.cdna4.buffer_load(ptr=x_ptr, offsets=x_offs, mask=m_mask[:, None])

    smem_rot = gl.allocate_shared_memory(gl.bfloat16, [RS, RS], layout=shared_rot)
    offs_rr = gl.arange(0, RS, layout=gl.SliceLayout(1, blocked_kn))
    offs_rc = gl.arange(0, RS, layout=gl.SliceLayout(0, blocked_kn))
    rot_data = gl.amd.cdna4.buffer_load(ptr=rot_ptr,
        offsets=offs_rr[:, None] * stride_rot_r + offs_rc[None, :] * stride_rot_c)
    smem_rot.store(rot_data)

    acc = gl.zeros((BLOCK_M, RS), gl.float32, layout=mfma_layout)
    acc = gl.amd.cdna4.mfma(gl.convert_layout(x_tile, dot_a),
                             smem_rot.load(layout=dot_b), acc)
    # Truncate to bf16 precision to match torch matmul (prevent decode drift)
    acc = acc.to(gl.bfloat16).to(gl.float32)

    # ======== Quantization DIRECTLY in MFMA layout ========
    # acc is [BLOCK_M, RS] in mfma_layout
    # Reshape to [BLOCK_M, NUM_QG, QG] — this reshape is free in MFMA layout
    # because tiles_per_warp=[1,4] and QG=32=2*16 spans exactly 2 tiles
    acc_g = gl.reshape(acc, (BLOCK_M, NUM_QG, QG))

    # Per-group abs max — operates on the MFMA-layout reshaped tensor
    # Reduction over last axis (32 elements per group)
    # In MFMA layout: each thread has 8 of 32, reduction across 4 lane-groups
    # Gluon's gl.max handles this via warp shuffles automatically
    amax = gl.max(gl.abs(acc_g), axis=-1)  # [BLOCK_M, NUM_QG]

    # Direct exponent extraction (round-up to match HIP fp4_scale carry logic)
    amax_u32 = amax.to(gl.uint32, bitcast=True)
    amax_u32 = (amax_u32 + 0x400000) & 0xFF800000
    raw_exp = (amax_u32 >> 23) & 0xFF
    e8m0 = gl.maximum(raw_exp, 2) - 2
    e8m0_u8 = e8m0.to(gl.uint8)

    # ======== Hardware FP4 conversion (gfx950 v_cvt_scalef32_pk_fp4_f32) ========
    # The hw instruction does: fp4 = round_e2m1(value / scale) with banker's rounding,
    # matching HIP vec_convert<fp4> exactly.
    # scale = 2^(e8m0 - 127) constructed from e8m0 bits.
    hw_scale = (e8m0.to(gl.uint32) << 23).to(gl.float32, bitcast=True)  # [BM, NUM_QG]

    # Broadcast scale to full quant-group shape via arithmetic
    # gl.expand_dims broadcasting works for binary ops (already proven with inv_scale)
    hw_scale_full = acc_g * 0.0 + gl.expand_dims(hw_scale, axis=2)  # [BM, NUM_QG, QG]

    # Split into adjacent pairs for the pk (pack-2) instruction
    acc_pairs = gl.reshape(acc_g, (BLOCK_M, NUM_QG, HALF_QG, 2))
    even_vals, odd_vals = gl.split(acc_pairs)  # each [BM, NUM_QG, HALF_QG]

    scale_pairs = gl.reshape(hw_scale_full, (BLOCK_M, NUM_QG, HALF_QG, 2))
    scale_even, _ = gl.split(scale_pairs)  # [BM, NUM_QG, HALF_QG] (same per pair)

    # Flatten for element-wise inline ASM
    even_flat = gl.reshape(even_vals, (BLOCK_M, NUM_QG * HALF_QG))
    odd_flat = gl.reshape(odd_vals, (BLOCK_M, NUM_QG * HALF_QG))
    scale_flat = gl.reshape(scale_even, (BLOCK_M, NUM_QG * HALF_QG))

    # Hardware FP4 conversion: 1 instruction per pair → packed fp4×2 in low byte
    packed_u32 = tl.inline_asm_elementwise(
        asm="v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3",
        constraints="=v,v,v,v",
        args=[even_flat, odd_flat, scale_flat],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    packed = packed_u32.to(gl.uint8)
    packed = gl.reshape(packed, (BLOCK_M, RS // 2))

    # ======== Store — need convert_layout only for store coalescing ========
    # FP4 output: [BLOCK_M, RS//2] = [32, 64]
    fp4_store: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4], threads_per_warp=[2, 32],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0],
    )
    packed_s = gl.convert_layout(packed, fp4_store)
    fp4_base = pid_rot * (RS // 2)
    s_m = m_base + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, fp4_store))
    s_mask = s_m < M
    fp4_c = fp4_base + gl.arange(0, RS // 2, layout=gl.SliceLayout(0, fp4_store))
    gl.amd.cdna4.buffer_store(stored_value=packed_s, ptr=fp4_ptr,
        offsets=s_m[:, None] * stride_fp4_m + fp4_c[None, :], mask=s_mask[:, None])

    # Scale output: [BLOCK_M, NUM_QG] = [32, 4]
    sc_store: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1], threads_per_warp=[32, 2],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0],
    )
    sc_s = gl.convert_layout(e8m0_u8, sc_store)
    sc_m = m_base + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, sc_store))
    sc_mask = sc_m < M
    sc_base = pid_rot * NUM_QG
    sc_c = sc_base + gl.arange(0, NUM_QG, layout=gl.SliceLayout(0, sc_store))
    gl.amd.cdna4.buffer_store(stored_value=sc_s, ptr=scale_ptr,
        offsets=sc_m[:, None] * stride_sc_m + sc_c[None, :], mask=sc_mask[:, None])


# ---------------------------------------------------------------------------
def fused_gluon_v13(x, rotation, rotation_size=128, fp4_out=None, scales_out=None):
    """
    Fused rotation + MXFP4 quantization.

    Args:
        x: [M, K] bf16 input tensor
        rotation: [rotation_size, rotation_size] bf16 rotation matrix
        rotation_size: rotation block size (default 128)
        fp4_out: optional pre-allocated [M, K//2] uint8 output buffer
        scales_out: optional pre-allocated [M, K//32] uint8 output buffer

    Returns:
        (fp4, scales) — uses pre-allocated buffers if provided, else allocates new
    """
    assert x.ndim == 2
    M, K = x.shape
    assert K % rotation_size == 0

    if fp4_out is None:
        fp4_out = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
    if scales_out is None:
        scales_out = torch.empty((M, K // 32), dtype=torch.uint8, device=x.device)

    grid = (triton.cdiv(M, 32), K // rotation_size)
    _fused_rot_quant_v13[grid](
        x, rotation, fp4_out, scales_out, M, K,
        x.stride(0), rotation.stride(0), rotation.stride(1),
        fp4_out.stride(0), scales_out.stride(0),
        RS=rotation_size, QG=32, BLOCK_M=32, NUM_WARPS=4, num_warps=4,
    )
    return fp4_out, scales_out


def unfused_ref(x, rotation, rotation_size=128):
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    M, K = x.shape
    x_r = x.reshape(M, K // rotation_size, rotation_size)
    x_r = torch.einsum("bgs,sk->bgk", x_r, rotation.to(x.dtype))
    return dynamic_mxfp4_quant(x_r.reshape(M, K))


# ---------------------------------------------------------------------------
def main():
    import sys; sys.path.insert(0, "/data/jiangyon/vllm_rotation")
    from fused_rotation_mxfp4_quant_v5 import fused_rotation_mxfp4_quant as v5
    from fused_rotation_mxfp4_gluon_v9 import fused_rotation_mxfp4_quant_gluon as v9
    from fused_rotation_mxfp4_gluon_v12 import fused_gluon_v12 as v12

    device, RS, N = "cuda:0", 128, 5000

    print("=" * 80)
    print("v13: Quant in MFMA layout — NO convert_layout before quant")
    print("=" * 80)

    # Correctness
    x = torch.randn(32, 5120, dtype=torch.bfloat16, device=device)
    rot = torch.linalg.qr(torch.randn(RS, RS, device=device))[0].to(torch.bfloat16)
    fp4_u, _ = unfused_ref(x, rot, RS)
    try:
        fp4_g, _ = fused_gluon_v13(x, rot, RS)
        n = (fp4_g != fp4_u).sum().item()
        print(f"  Correctness: fp4_diff={n} ({n/fp4_u.numel()*100:.1f}%)")
    except Exception as e:
        print(f"  ❌ {type(e).__name__}: {str(e)[:500]}")
        return

    # Benchmark
    def bench(fn, M, K):
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        rot = torch.linalg.qr(torch.randn(RS, RS, device=device))[0].to(torch.bfloat16)
        for _ in range(500): _ = fn(x, rot, RS)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N): _ = fn(x, rot, RS)
        torch.cuda.synchronize()
        return (time.perf_counter()-t0)/N*1e6

    print(f"\n{'shape':>12s}  {'triton_v5':>10s}  {'gluon_v9':>10s}  {'gluon_v12':>10s}  {'gluon_v13':>10s}")
    for M, K in [(1,5120), (32,5120), (128,5120), (512,5120), (1024,5120), (1,51200), (32,51200)]:
        t5 = bench(v5, M, K)
        t9 = bench(v9, M, K)
        t12 = bench(v12, M, K)
        t13 = bench(fused_gluon_v13, M, K)
        print(f"  {M}x{K:>5d}  {t5:6.1f}µs({t5/t13:.2f})  {t9:6.1f}µs({t9/t13:.2f})  "
              f"{t12:6.1f}µs({t12/t13:.2f})  {t13:6.1f}µs(1.00)")


if __name__ == "__main__":
    main()
