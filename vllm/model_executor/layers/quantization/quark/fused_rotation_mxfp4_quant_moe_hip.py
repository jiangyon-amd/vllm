#!/usr/bin/env python3
"""
MoE Fused Rotation + MXFP4 Quant + Sorted Scale Scatter — HIP kernel.

Uses @triton.jit with tl.dot for rotation matmul and v_cvt_scalef32_pk_fp4_f32
ISA for FP4 quantization. No Gluon dependency — avoids gl.convert_layout
overhead that limits the gluon version.

Compared to the existing MoE Triton kernel:
  - v_cvt ISA replaces manual bit-manipulation FP4 quantization
  - Configurable BLOCK_M and num_warps for decode vs prefill
  - Specialized K=2048 topk=8 variant with hardcoded constants
"""

import torch
import triton
from triton import language as tl


@triton.jit
def _fused_decode_m1_rot_quant_sorted_hip_kernel(
    x_ptr,
    rot_ptr,
    fp4_ptr,
    sorted_ids_ptr,
    num_valid_ids_ptr,
    sorted_scale_ptr,
    stride_x_n,
    stride_rot_r,
    stride_rot_c,
    stride_fp4_m,
    stride_sc_m,
    stride_sc_n,
    token_num,
    n_i,
    tile_n,
    m_o,
    RS: tl.constexpr,
    QG: tl.constexpr,
    MAX_Q: tl.constexpr,
):
    pid_k = tl.program_id(0)
    NUM_QG: tl.constexpr = RS // QG
    HALF_QG: tl.constexpr = QG // 2
    col_base = pid_k * NUM_QG

    # ======== Load x [1, RS] and rotation [RS, RS] ========
    offs_k = tl.arange(0, RS)
    x = tl.load(x_ptr + (pid_k * RS + offs_k) * stride_x_n).to(tl.float32)
    offs_r = tl.arange(0, RS)
    offs_c = tl.arange(0, RS)
    rot = tl.load(
        rot_ptr + offs_r[:, None] * stride_rot_r + offs_c[None, :] * stride_rot_c
    ).to(tl.float32)

    # ======== Rotation matmul via tl.dot ========
    acc = tl.dot(x[None, :].to(tl.bfloat16), rot.to(tl.bfloat16))
    acc = acc.to(tl.bfloat16).to(tl.float32)

    # ======== MXFP4 quantization using v_cvt ISA ========
    acc_g = tl.reshape(acc, (1, NUM_QG, QG))
    amax = tl.max(tl.abs(acc_g), axis=-1, keep_dims=False)

    # MoE scale derivation (0x200000 rounding to match aiter separated path)
    amax_u32 = amax.to(tl.uint32, bitcast=True)
    amax_u32 = (amax_u32 + 0x200000) & 0xFF800000
    raw_exp = (amax_u32 >> 23) & 0xFF
    e8m0 = tl.maximum(raw_exp, 2) - 2
    e8m0_u8 = e8m0.to(tl.uint8)

    # hw_scale for v_cvt_scalef32_pk_fp4_f32 (divides input by this value)
    hw_scale = (e8m0.to(tl.uint32) << 23).to(tl.float32, bitcast=True)

    # Reshape to pairs for v_cvt
    acc_pairs = tl.reshape(acc_g, (1, NUM_QG, HALF_QG, 2))
    even_vals, odd_vals = tl.split(acc_pairs)
    even_flat = tl.reshape(even_vals, (1, NUM_QG * HALF_QG))
    odd_flat = tl.reshape(odd_vals, (1, NUM_QG * HALF_QG))

    hw_scale_broad = tl.broadcast_to(
        tl.reshape(hw_scale, (1, NUM_QG, 1)),
        (1, NUM_QG, HALF_QG),
    )
    scale_flat = tl.reshape(hw_scale_broad, (1, NUM_QG * HALF_QG))

    packed_u32 = tl.inline_asm_elementwise(
        asm="v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3",
        constraints="=v,v,v,v",
        args=[even_flat, odd_flat, scale_flat],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    packed = packed_u32.to(tl.uint8)
    packed = tl.reshape(packed, (RS // 2,))

    # ======== Store FP4 (row 0) ========
    fp4_offs = pid_k * (RS // 2) + tl.arange(0, RS // 2)
    tl.store(fp4_ptr + fp4_offs, packed)

    # ======== Sorted scale scatter (2D broadcast + scatter) ========
    num_valid = tl.load(num_valid_ids_ptr)
    q = tl.arange(0, MAX_Q)
    sid = tl.load(sorted_ids_ptr + q, mask=q < num_valid, other=token_num)
    tok = sid & 0xFFFFFF
    q_valid = (q < num_valid) & (tok < token_num)

    cols = col_base + tl.arange(0, NUM_QG)
    col_valid = cols < n_i
    vals_row = tl.reshape(e8m0_u8, (NUM_QG,))
    vals = tl.broadcast_to(vals_row[None, :], (MAX_Q, NUM_QG))
    vals = tl.where(q_valid[:, None], vals, 0)

    pid_m_out = q // 32
    q_in = q % 32
    q_half = q_in // 16
    m_local = q_in % 16
    pid_n = cols // 8
    n_local = cols % 4
    half = (cols % 8) // 4
    i = q_half[:, None] + 2 * half[None, :]
    flat = (
        (((pid_m_out[:, None] * tile_n + pid_n[None, :]) * 4 + n_local[None, :]) * 16
         + m_local[:, None]) * 4
        + i
    )
    dst_row = flat // n_i
    dst_col = flat % n_i

    mask = col_valid[None, :] & (q[:, None] < m_o)
    tl.store(
        sorted_scale_ptr + dst_row * stride_sc_m + dst_col * stride_sc_n,
        vals,
        mask=mask,
    )


@triton.jit
def _fused_decode_m1_topk8_k2048_hip_kernel(
    x_ptr,
    rot_ptr,
    fp4_ptr,
    sorted_ids_ptr,
    num_valid_ids_ptr,
    sorted_scale_ptr,
    stride_x_n,
    stride_rot_r,
    stride_rot_c,
    stride_fp4_m,
    stride_sc_m,
    stride_sc_n,
    token_num,
    m_o,
    MAX_Q: tl.constexpr,
):
    """Specialized for K=2048, RS=128, QG=32, n_i=64, topk=8."""
    RS: tl.constexpr = 128
    QG: tl.constexpr = 32
    NUM_QG: tl.constexpr = 4
    HALF_QG: tl.constexpr = 16
    pid_k = tl.program_id(0)
    col_base = pid_k * NUM_QG

    offs_k = tl.arange(0, RS)
    x = tl.load(x_ptr + (pid_k * RS + offs_k) * stride_x_n).to(tl.float32)
    offs_r = tl.arange(0, RS)
    offs_c = tl.arange(0, RS)
    rot = tl.load(
        rot_ptr + offs_r[:, None] * stride_rot_r + offs_c[None, :] * stride_rot_c
    ).to(tl.float32)

    acc = tl.dot(x[None, :].to(tl.bfloat16), rot.to(tl.bfloat16))
    acc = acc.to(tl.bfloat16).to(tl.float32)

    acc_g = tl.reshape(acc, (1, NUM_QG, QG))
    amax = tl.max(tl.abs(acc_g), axis=-1, keep_dims=False)

    amax_u32 = amax.to(tl.uint32, bitcast=True)
    amax_u32 = (amax_u32 + 0x200000) & 0xFF800000
    raw_exp = (amax_u32 >> 23) & 0xFF
    e8m0 = tl.maximum(raw_exp, 2) - 2
    e8m0_u8 = e8m0.to(tl.uint8)

    hw_scale = (e8m0.to(tl.uint32) << 23).to(tl.float32, bitcast=True)

    acc_pairs = tl.reshape(acc_g, (1, NUM_QG, HALF_QG, 2))
    even_vals, odd_vals = tl.split(acc_pairs)
    even_flat = tl.reshape(even_vals, (1, NUM_QG * HALF_QG))
    odd_flat = tl.reshape(odd_vals, (1, NUM_QG * HALF_QG))

    hw_scale_broad = tl.broadcast_to(
        tl.reshape(hw_scale, (1, NUM_QG, 1)),
        (1, NUM_QG, HALF_QG),
    )
    scale_flat = tl.reshape(hw_scale_broad, (1, NUM_QG * HALF_QG))

    packed_u32 = tl.inline_asm_elementwise(
        asm="v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3",
        constraints="=v,v,v,v",
        args=[even_flat, odd_flat, scale_flat],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    packed = packed_u32.to(tl.uint8)
    packed = tl.reshape(packed, (RS // 2,))

    fp4_offs = pid_k * (RS // 2) + tl.arange(0, RS // 2)
    tl.store(fp4_ptr + fp4_offs, packed)

    # Sorted scale scatter — hardcoded for n_i=64, tile_n=8
    num_valid = tl.load(num_valid_ids_ptr)
    q = tl.arange(0, MAX_Q)
    sid = tl.load(sorted_ids_ptr + q, mask=q < num_valid, other=token_num)
    tok = sid & 0xFFFFFF
    q_valid = (q < num_valid) & (tok < token_num)

    cols = col_base + tl.arange(0, NUM_QG)
    vals_row = tl.reshape(e8m0_u8, (NUM_QG,))
    vals = tl.broadcast_to(vals_row[None, :], (MAX_Q, NUM_QG))
    vals = tl.where(q_valid[:, None], vals, 0)

    pid_m_out = q >> 5
    q_in = q & 31
    q_half = q_in >> 4
    m_local = q_in & 15
    pid_n = cols >> 3
    n_local = cols & 3
    half = (cols & 7) >> 2
    i = q_half[:, None] + (half[None, :] << 1)

    flat = (
        ((((pid_m_out[:, None] << 3) + pid_n[None, :]) << 2) + n_local[None, :]) * 16
        + m_local[:, None]
    ) * 4 + i
    dst_row = flat >> 6
    dst_col = flat & 63

    mask = q[:, None] < m_o
    tl.store(
        sorted_scale_ptr + dst_row * stride_sc_m + dst_col * stride_sc_n,
        vals,
        mask=mask,
    )


def _pick_decode_max_q(m_pad: int) -> int:
    if m_pad <= 64:
        return 64
    elif m_pad <= 128:
        return 128
    else:
        return 256


def fused_decode_m1_rot_quant_sorted_hip(
    x, rotation, fp4_out, sorted_ids, num_valid_ids, sorted_scale_out,
    K, RS, QG, n_i, token_num, m_o, m_pad,
):
    """Launch HIP MoE decode kernel."""
    tile_n = triton.cdiv(n_i, 8)
    max_q = _pick_decode_max_q(m_pad)

    grid = (K // RS,)

    use_topk8_special = (
        K == 2048 and RS == 128 and n_i == 64 and max_q <= 256
    )

    if use_topk8_special:
        _fused_decode_m1_topk8_k2048_hip_kernel[grid](
            x, rotation, fp4_out,
            sorted_ids, num_valid_ids, sorted_scale_out,
            x.stride(1),
            rotation.stride(0), rotation.stride(1),
            fp4_out.stride(0),
            sorted_scale_out.stride(0), sorted_scale_out.stride(1),
            token_num=token_num, m_o=m_o,
            MAX_Q=max_q,
            num_warps=4,
        )
    else:
        _fused_decode_m1_rot_quant_sorted_hip_kernel[grid](
            x, rotation, fp4_out,
            sorted_ids, num_valid_ids, sorted_scale_out,
            x.stride(1),
            rotation.stride(0), rotation.stride(1),
            fp4_out.stride(0),
            sorted_scale_out.stride(0), sorted_scale_out.stride(1),
            token_num=token_num, n_i=n_i,
            tile_n=tile_n, m_o=m_o,
            RS=RS, QG=QG, MAX_Q=max_q,
            num_warps=4,
        )
    return fp4_out, sorted_scale_out
