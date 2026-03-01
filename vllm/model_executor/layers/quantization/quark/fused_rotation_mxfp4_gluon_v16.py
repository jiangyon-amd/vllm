#!/usr/bin/env python3
"""
Fused Rotation + MXFP4 Quant v16 — Unified Gluon kernel for all M

Single kernel supporting:
  - All M values (decode M=1 to prefill M=2048+)
  - Optional fused e8m0_shuffle (compile-time constexpr)
  - Pre-allocated output buffers
  - MFMA layout direct quantization (no convert_layout before quant)
  - Hardware v_cvt_scalef32_pk_fp4_f32 instruction
"""

import torch
import triton
from triton import language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import time


@gluon.jit
def _fused_rot_quant_unified(
    x_ptr, rot_ptr, fp4_ptr, scale_ptr,
    M, K, sn_padded,
    stride_x_m, stride_rot_r, stride_rot_c,
    stride_fp4_m, stride_sc_m,
    RS: gl.constexpr,
    QG: gl.constexpr,
    BLOCK_M: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SHUFFLE_SCALES: gl.constexpr,
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

    # ======== Rotation matmul ========
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

    # ======== Quantization in MFMA layout ========
    acc_g = gl.reshape(acc, (BLOCK_M, NUM_QG, QG))
    amax = gl.max(gl.abs(acc_g), axis=-1)

    amax_u32 = amax.to(gl.uint32, bitcast=True)
    amax_u32 = (amax_u32 + 0x400000) & 0xFF800000
    raw_exp = (amax_u32 >> 23) & 0xFF
    e8m0 = gl.maximum(raw_exp, 2) - 2
    e8m0_u8 = e8m0.to(gl.uint8)

    hw_scale = (e8m0.to(gl.uint32) << 23).to(gl.float32, bitcast=True)
    hw_scale_full = acc_g * 0.0 + gl.expand_dims(hw_scale, axis=2)

    acc_pairs = gl.reshape(acc_g, (BLOCK_M, NUM_QG, HALF_QG, 2))
    even_vals, odd_vals = gl.split(acc_pairs)
    scale_pairs = gl.reshape(hw_scale_full, (BLOCK_M, NUM_QG, HALF_QG, 2))
    scale_even, _ = gl.split(scale_pairs)

    even_flat = gl.reshape(even_vals, (BLOCK_M, NUM_QG * HALF_QG))
    odd_flat = gl.reshape(odd_vals, (BLOCK_M, NUM_QG * HALF_QG))
    scale_flat = gl.reshape(scale_even, (BLOCK_M, NUM_QG * HALF_QG))

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

    # ======== Store FP4 ========
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

    # ======== Store Scales ========
    sc_store: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1], threads_per_warp=[32, 2],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0],
    )
    sc_s = gl.convert_layout(e8m0_u8, sc_store)

    if SHUFFLE_SCALES:
        # Fused e8m0_shuffle store
        sc_m_local = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, sc_store))
        sc_q = gl.arange(0, NUM_QG, layout=gl.SliceLayout(0, sc_store))
        i1 = sc_m_local // 16
        i2 = sc_m_local % 16
        orig_col = pid_rot * NUM_QG + sc_q
        i3 = orig_col // 8
        i4 = (orig_col % 8) // 4
        i5 = orig_col % 4
        flat_idx = i3[None, :] * 256 + i5[None, :] * 64 + i2[:, None] * 4 + i4[None, :] * 2 + i1[:, None]
        sh_row = flat_idx // sn_padded + m_base
        sh_col = flat_idx % sn_padded
        sc_mask_sh = sh_row < M
        gl.amd.cdna4.buffer_store(stored_value=sc_s, ptr=scale_ptr,
            offsets=sh_row * stride_sc_m + sh_col, mask=sc_mask_sh)
    else:
        # Direct store (no shuffle)
        sc_m = m_base + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, sc_store))
        sc_mask = sc_m < M
        sc_base = pid_rot * NUM_QG
        sc_c = sc_base + gl.arange(0, NUM_QG, layout=gl.SliceLayout(0, sc_store))
        gl.amd.cdna4.buffer_store(stored_value=sc_s, ptr=scale_ptr,
            offsets=sc_m[:, None] * stride_sc_m + sc_c[None, :], mask=sc_mask[:, None])


def fused_gluon_unified(x, rotation, rotation_size=128, fp4_out=None, scales_out=None, shuffle_scales=False):
    """
    Unified fused rotation+MXFP4 quant for all M.
    
    Args:
        shuffle_scales: If True, apply e8m0_shuffle in-kernel (for M>=32 GEMM compatibility)
    """
    assert x.ndim == 2
    M, K = x.shape
    assert K % rotation_size == 0

    QGROUP = 32
    n_scales = K // QGROUP

    if fp4_out is None:
        fp4_out = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)

    if shuffle_scales:
        sn_padded = (n_scales + 7) // 8 * 8
        sm_padded = (M + 255) // 256 * 256
        if scales_out is None:
            scales_out = torch.zeros((sm_padded, sn_padded), dtype=torch.uint8, device=x.device)
        else:
            if scales_out.shape[0] < sm_padded or scales_out.shape[1] < sn_padded:
                scales_out = torch.zeros((sm_padded, sn_padded), dtype=torch.uint8, device=x.device)
            else:
                scales_out.zero_()
    else:
        sn_padded = n_scales
        if scales_out is None:
            scales_out = torch.empty((M, n_scales), dtype=torch.uint8, device=x.device)

    grid = (triton.cdiv(M, 32), K // rotation_size)
    _fused_rot_quant_unified[grid](
        x, rotation, fp4_out, scales_out, M, K, sn_padded,
        x.stride(0), rotation.stride(0), rotation.stride(1),
        fp4_out.stride(0), scales_out.stride(0),
        RS=rotation_size, QG=32, BLOCK_M=32, NUM_WARPS=4, num_warps=4,
        SHUFFLE_SCALES=shuffle_scales,
    )
    return fp4_out, scales_out


# ======== Unit test ========
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/data/jiangyon/vllm_rotation')
    from aiter import per_1x32_f4_quant_hip
    from aiter.utility.fp4_utils import e8m0_shuffle
    from fused_rotation_mxfp4_gluon_v13 import fused_gluon_v13 as v13

    device = "cuda:0"
    RS = 128
    WARMUP, N_ITER = 500, 5000

    print("=" * 80)
    print("v16 Unified Gluon — Correctness & Performance")
    print("=" * 80)

    rot = torch.linalg.qr(torch.randn(RS, RS, device=device))[0].to(torch.bfloat16)

    # === Correctness ===
    print("\n--- Correctness (no shuffle) ---")
    for M, K in [(1, 3584), (4, 3584), (32, 3584), (128, 3584), (32, 18944)]:
        torch.manual_seed(42)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        fp4_ref, sc_ref = v13(x, rot, RS)
        fp4_v16, sc_v16 = fused_gluon_unified(x, rot, RS, shuffle_scales=False)
        match = torch.equal(fp4_v16, fp4_ref) and torch.equal(sc_v16, sc_ref)
        print(f"  {'✅' if match else '❌'} M={M:>3d} K={K}: match={match}", flush=True)

    print("\n--- Correctness (with shuffle, M>=32) ---")
    for M, K in [(32, 3584), (64, 3584), (128, 3584), (32, 18944)]:
        torch.manual_seed(42)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        # Reference: v13 + external e8m0_shuffle
        fp4_ref, sc_ref_raw = v13(x, rot, RS)
        sc_ref = e8m0_shuffle(sc_ref_raw)
        # v16 fused shuffle
        fp4_v16, sc_v16 = fused_gluon_unified(x, rot, RS, shuffle_scales=True)
        fp4_match = torch.equal(fp4_v16, fp4_ref)
        sc_match = torch.equal(sc_v16[:M, :K//32], sc_ref[:M, :K//32])
        print(f"  {'✅' if fp4_match and sc_match else '❌'} M={M:>3d} K={K}: fp4={fp4_match} sc={sc_match}", flush=True)

    # === Performance ===
    print("\n--- Performance ---")

    def bench(fn, *args):
        for _ in range(WARMUP):
            fn(*args)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            fn(*args)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / N_ITER * 1e6

    def separated_no_shuffle(x, rot, rs):
        M, K = x.shape
        x_r = (x.reshape(M, K // rs, rs) @ rot).reshape(M, K)
        return per_1x32_f4_quant_hip(x_r, shuffle=False)

    def separated_shuffle(x, rot, rs):
        M, K = x.shape
        x_r = (x.reshape(M, K // rs, rs) @ rot).reshape(M, K)
        return per_1x32_f4_quant_hip(x_r, shuffle=True)

    print(f"{'Shape':<15s}  {'Separated':>10s}  {'v16':>8s}  {'v16_pre':>8s}  {'Speedup':>8s}  {'Note':>12s}")
    print("-" * 75)

    for M, K in [(1, 3584), (4, 3584), (16, 3584), (32, 3584), (64, 3584), (128, 3584), (256, 3584), (32, 18944), (128, 18944)]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)

        if M < 32:
            t_sep = bench(separated_no_shuffle, x, rot, RS)
            fp4 = torch.empty(M, K // 2, dtype=torch.uint8, device=device)
            sc = torch.empty(M, K // 32, dtype=torch.uint8, device=device)
            t_v16 = bench(fused_gluon_unified, x, rot, RS)
            t_v16p = bench(fused_gluon_unified, x, rot, RS, fp4, sc, False)
            note = "no shuffle"
        else:
            t_sep = bench(separated_shuffle, x, rot, RS)
            sn_pad = (K // 32 + 7) // 8 * 8
            sm_pad = (M + 255) // 256 * 256
            fp4 = torch.empty(M, K // 2, dtype=torch.uint8, device=device)
            sc_pad = torch.zeros(sm_pad, sn_pad, dtype=torch.uint8, device=device)
            t_v16 = bench(fused_gluon_unified, x, rot, RS, None, None, True)
            t_v16p = bench(fused_gluon_unified, x, rot, RS, fp4, sc_pad, True)
            note = "fused shuffle"

        sp = t_sep / t_v16p
        print(f"  {M:>4d}x{K:<5d}  {t_sep:8.1f}us  {t_v16:6.1f}us  {t_v16p:6.1f}us  {sp:6.2f}x  {note:>12s}", flush=True)

    # Per-token summary
    print(f"\n--- Per-Token Summary (36 layers × 4 linears) ---")
    for M in [1, 32, 128]:
        total_sep = 0
        total_v16 = 0
        for K in [3584, 3584, 18944, 3584]:
            x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            if M < 32:
                total_sep += bench(separated_no_shuffle, x, rot, RS) * 36
                fp4 = torch.empty(M, K // 2, dtype=torch.uint8, device=device)
                sc = torch.empty(M, K // 32, dtype=torch.uint8, device=device)
                total_v16 += bench(fused_gluon_unified, x, rot, RS, fp4, sc, False) * 36
            else:
                total_sep += bench(separated_shuffle, x, rot, RS) * 36
                sn_pad = (K // 32 + 7) // 8 * 8
                sm_pad = (M + 255) // 256 * 256
                fp4 = torch.empty(M, K // 2, dtype=torch.uint8, device=device)
                sc_pad = torch.zeros(sm_pad, sn_pad, dtype=torch.uint8, device=device)
                total_v16 += bench(fused_gluon_unified, x, rot, RS, fp4, sc_pad, True) * 36
        print(f"  M={M:>3d}: Sep={total_sep/1000:.2f}ms  v16={total_v16/1000:.2f}ms  "
              f"Speedup={total_sep/total_v16:.2f}x  Save={( total_sep-total_v16)/1000:.2f}ms", flush=True)
