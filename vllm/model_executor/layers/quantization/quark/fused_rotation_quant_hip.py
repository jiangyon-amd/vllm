#!/usr/bin/env python3
"""
Fused Rotation + MXFP4 Quant — HIP C++ kernel via Triton inline ASM

Uses MFMA instructions directly for rotation matmul,
v_cvt_scalef32_pk_fp4_f32 for FP4 quantization,
and direct global stores for shuffled scales (no convert_layout overhead).

This is implemented as a Triton kernel using tl.inline_asm_elementwise
for critical HIP instructions, avoiding Gluon's layout constraints.
"""

import torch
import triton
from triton import language as tl


@triton.jit
def _fused_rot_quant_hip_kernel(
    x_ptr, rot_ptr, fp4_ptr, scale_ptr,
    M, K, sn_padded,
    stride_x_m, stride_rot_r, stride_rot_c,
    stride_fp4_m, stride_sc_m,
    RS: tl.constexpr,
    QG: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SHUFFLE_SCALES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_rot = tl.program_id(1)

    NUM_QG: tl.constexpr = RS // QG
    HALF_QG: tl.constexpr = QG // 2

    m_base = pid_m * BLOCK_M
    col_base = pid_rot * RS

    # ======== Load x [BLOCK_M, RS] ========
    offs_m = m_base + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, RS)
    m_mask = offs_m < M

    x = tl.load(
        x_ptr + offs_m[:, None] * stride_x_m + (col_base + offs_k)[None, :],
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    # ======== Load rotation [RS, RS] ========
    rot = tl.load(
        rot_ptr + offs_k[:, None] * stride_rot_r + tl.arange(0, RS)[None, :] * stride_rot_c,
    ).to(tl.float32)

    # ======== Rotation matmul ========
    # x [BLOCK_M, RS] @ rot [RS, RS] → acc [BLOCK_M, RS]
    acc = tl.dot(x, rot)

    # Truncate to bf16 precision to match torch matmul
    acc = acc.to(tl.bfloat16).to(tl.float32)

    # ======== Quantization ========
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr = QG
    NUM_QUANT_BLOCKS: tl.constexpr = RS // MXFP4_QUANT_BLOCK_SIZE

    acc_g = tl.reshape(acc, (BLOCK_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE))
    amax = tl.max(tl.abs(acc_g), axis=-1)

    # Scale computation (matching HIP ASM fp4_scale carry logic: 0x400000)
    amax_u32 = amax.to(tl.uint32, bitcast=True)
    amax_u32 = (amax_u32 + 0x400000) & 0xFF800000
    raw_exp = (amax_u32 >> 23) & 0xFF
    e8m0 = tl.maximum(raw_exp, 2) - 2
    e8m0_u8 = e8m0.to(tl.uint8)

    # Construct hw_scale for v_cvt_scalef32_pk_fp4_f32
    hw_scale = (e8m0.to(tl.uint32) << 23).to(tl.float32, bitcast=True)

    # Reshape acc to pairs for v_cvt_scalef32_pk_fp4_f32
    acc_pairs = tl.reshape(acc_g, (BLOCK_M, NUM_QUANT_BLOCKS, HALF_QG, 2))
    even_vals, odd_vals = tl.split(acc_pairs)
    even_flat = tl.reshape(even_vals, (BLOCK_M, NUM_QUANT_BLOCKS * HALF_QG))
    odd_flat = tl.reshape(odd_vals, (BLOCK_M, NUM_QUANT_BLOCKS * HALF_QG))

    # Broadcast scale: [BLOCK_M, NUM_QG] → [BLOCK_M, NUM_QG, HALF_QG] → flat
    hw_scale_broad = tl.broadcast_to(
        tl.reshape(hw_scale, (BLOCK_M, NUM_QUANT_BLOCKS, 1)),
        (BLOCK_M, NUM_QUANT_BLOCKS, HALF_QG)
    )
    scale_flat = tl.reshape(hw_scale_broad, (BLOCK_M, NUM_QUANT_BLOCKS * HALF_QG))

    # v_cvt_scalef32_pk_fp4_f32: pack 2 fp32 → 1 byte fp4x2
    packed_u32 = tl.inline_asm_elementwise(
        asm="v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3",
        constraints="=v,v,v,v",
        args=[even_flat, odd_flat, scale_flat],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    packed = packed_u32.to(tl.uint8)
    packed = tl.reshape(packed, (BLOCK_M, RS // 2))

    # ======== Store FP4 ========
    fp4_base = pid_rot * (RS // 2)
    fp4_offs = offs_m[:, None] * stride_fp4_m + (fp4_base + tl.arange(0, RS // 2))[None, :]
    tl.store(fp4_ptr + fp4_offs, packed, mask=m_mask[:, None])

    # ======== Store Scales ========
    if SHUFFLE_SCALES:
        # Shuffle store: scatter e8m0 to shuffled positions
        # No convert_layout needed — direct from registers to global
        sc_m_local = tl.arange(0, BLOCK_M)
        sc_q = tl.arange(0, NUM_QG)
        i1 = sc_m_local // 16
        i2 = sc_m_local % 16
        orig_col = pid_rot * NUM_QG + sc_q
        i3 = orig_col // 8
        i4 = (orig_col % 8) // 4
        i5 = orig_col % 4
        flat_idx = (i3[None, :] * 256 + i5[None, :] * 64
                    + i2[:, None] * 4 + i4[None, :] * 2 + i1[:, None])
        sh_row = flat_idx // sn_padded + m_base
        sh_col = flat_idx % sn_padded
        sc_mask_sh = (sh_row < M) & m_mask[:, None]
        sc_offsets = sh_row * stride_sc_m + sh_col
        tl.store(scale_ptr + sc_offsets, e8m0_u8, mask=sc_mask_sh)
    else:
        # Direct store
        sc_base = pid_rot * NUM_QG
        sc_offs = offs_m[:, None] * stride_sc_m + (sc_base + tl.arange(0, NUM_QG))[None, :]
        tl.store(scale_ptr + sc_offs, e8m0_u8, mask=m_mask[:, None])


def fused_rotation_quant_hip(x, rotation, rotation_size=128, fp4_out=None, scales_out=None, shuffle_scales=False):
    """
    Fused rotation+MXFP4 quant using Triton with HIP inline ASM.
    No Gluon dependency — pure Triton with tl.dot and tl.inline_asm.
    """
    assert x.ndim == 2
    M, K = x.shape
    assert K % rotation_size == 0

    RS = rotation_size
    QGROUP = 32
    n_scales = K // QGROUP

    if fp4_out is None:
        fp4_out = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)

    if shuffle_scales:
        sn_padded = (n_scales + 7) // 8 * 8
        sm_padded = (M + 255) // 256 * 256
        if scales_out is None:
            scales_out = torch.empty((sm_padded, sn_padded), dtype=torch.uint8, device=x.device)
    else:
        sn_padded = n_scales
        if scales_out is None:
            scales_out = torch.empty((M, n_scales), dtype=torch.uint8, device=x.device)

    # Decode/small-batch path benefits from smaller tiles.
    if M <= 4:
        BLOCK_M = 4
        NUM_WARPS = 1
        NUM_STAGES = 2
    elif M <= 16:
        BLOCK_M = 16
        NUM_WARPS = 2
        NUM_STAGES = 2
    else:
        BLOCK_M = 32
        NUM_WARPS = 4
        NUM_STAGES = 3

    grid = (triton.cdiv(M, BLOCK_M), K // RS)

    _fused_rot_quant_hip_kernel[grid](
        x, rotation, fp4_out, scales_out,
        M, K, sn_padded,
        x.stride(0), rotation.stride(0), rotation.stride(1),
        fp4_out.stride(0), scales_out.stride(0),
        RS=RS, QG=QGROUP, BLOCK_M=BLOCK_M,
        SHUFFLE_SCALES=shuffle_scales,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    return fp4_out, scales_out


if __name__ == "__main__":
    import time
    from aiter import per_1x32_f4_quant_hip

    device = "cuda:0"
    RS = 128
    WARMUP = 500
    N_ITER = 5000

    rot = torch.linalg.qr(torch.randn(RS, RS, device=device))[0].to(torch.bfloat16)

    def bench(fn):
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / N_ITER * 1e6

    K = 3584
    print(f"{'M':>4s}  {'Sep':>8s}  {'HIP_fused':>10s}  {'Speedup':>8s}  {'fp4%':>6s}")
    print("-" * 45)

    for M in [1, 4, 32, 128]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        shuffle = True

        t_sep = bench(lambda: per_1x32_f4_quant_hip(
            (x.reshape(M, K // RS, RS) @ rot).reshape(M, K), shuffle=shuffle))

        sn = (K // 32 + 7) // 8 * 8
        sm = (M + 255) // 256 * 256
        fp4 = torch.empty(M, K // 2, dtype=torch.uint8, device=device)
        sc = torch.empty(sm, sn, dtype=torch.uint8, device=device)
        t_hip = bench(lambda: fused_rotation_quant_hip(
            x, rot, RS, fp4_out=fp4, scales_out=sc, shuffle_scales=shuffle))

        # Correctness
        x_rot = (x.reshape(M, K // RS, RS) @ rot).reshape(M, K)
        q_ref, _ = per_1x32_f4_quant_hip(x_rot, shuffle=shuffle)
        q_f, _ = fused_rotation_quant_hip(x, rot, RS, fp4_out=fp4, scales_out=sc, shuffle_scales=shuffle)
        fp4_m = (q_ref.view(torch.uint8)[:M] == q_f[:M]).float().mean() * 100

        print(f"  {M:>3d}  {t_sep:6.1f}us  {t_hip:8.1f}us  {t_sep/t_hip:.2f}x  {fp4_m:.1f}%", flush=True)
