#!/usr/bin/env python3
"""
Triton MoE 3-in-1: Fused Rotation + MXFP4 Quant + Sorted Scale Scatter (M>=1)

Pure Triton kernel (no Gluon). Extends the M=1 topk8 kernel to M>1 prefill.
Advantages over Gluon 3-in-1: no gl.convert_layout overhead.
"""

import torch
import triton
from triton import language as tl


@triton.jit
def _fused_rot_quant_sort_triton_kernel(
    x_ptr, rot_ptr, fp4_ptr,
    sorted_ids_ptr, num_valid_ids_ptr, sorted_scale_ptr,
    M, K, token_num, m_pad,
    stride_x_m, stride_x_n,
    stride_rot_r, stride_rot_c,
    stride_fp4_m,
    stride_sc_m, stride_sc_n,
    RS: tl.constexpr,
    QG: tl.constexpr,
    BLOCK_M: tl.constexpr,
    N_I: tl.constexpr,
    TILE_N: tl.constexpr,
    MAX_Q: tl.constexpr,
):
    NUM_QG: tl.constexpr = RS // QG
    HALF_QG: tl.constexpr = QG // 2

    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    col_base = pid_k * NUM_QG
    m_base = pid_m * BLOCK_M

    # Load rotation matrix [RS, RS] (shared across all rows)
    offs_r = tl.arange(0, RS)
    offs_c = tl.arange(0, RS)
    rot = tl.load(
        rot_ptr + offs_r[:, None] * stride_rot_r + offs_c[None, :] * stride_rot_c
    ).to(tl.float32)

    # Process BLOCK_M rows
    offs_m = tl.arange(0, BLOCK_M)
    m_idx = m_base + offs_m
    m_mask = m_idx < M

    # Load x [BLOCK_M, RS]
    offs_k = tl.arange(0, RS)
    x = tl.load(
        x_ptr + m_idx[:, None] * stride_x_m + (pid_k * RS + offs_k)[None, :] * stride_x_n,
        mask=m_mask[:, None], other=0.0
    ).to(tl.float32)

    # Rotation matmul: x[BLOCK_M, RS] @ rot[RS, RS] -> acc[BLOCK_M, RS]
    acc = tl.dot(x.to(tl.bfloat16), rot.to(tl.bfloat16))
    acc = acc.to(tl.bfloat16).to(tl.float32)

    # MXFP4 quantization per row
    acc_g = tl.reshape(acc, (BLOCK_M, NUM_QG, QG))
    amax = tl.max(tl.abs(acc_g), axis=-1, keep_dims=False)

    amax_u32 = amax.to(tl.uint32, bitcast=True)
    amax_u32 = (amax_u32 + 0x200000) & 0xFF800000
    raw_exp = (amax_u32 >> 23) & 0xFF
    e8m0 = tl.maximum(raw_exp, 2) - 2
    e8m0_u8 = e8m0.to(tl.uint8)

    hw_scale = (e8m0.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    hw_scale_broad = tl.broadcast_to(
        tl.reshape(hw_scale, (BLOCK_M, NUM_QG, 1)), (BLOCK_M, NUM_QG, HALF_QG)
    )

    acc_pairs = tl.reshape(acc_g, (BLOCK_M, NUM_QG, HALF_QG, 2))
    even_vals, odd_vals = tl.split(acc_pairs)
    even_flat = tl.reshape(even_vals, (BLOCK_M, NUM_QG * HALF_QG))
    odd_flat = tl.reshape(odd_vals, (BLOCK_M, NUM_QG * HALF_QG))
    scale_flat = tl.reshape(hw_scale_broad, (BLOCK_M, NUM_QG * HALF_QG))

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

    # Store FP4
    fp4_offs_m = m_idx
    fp4_offs_n = pid_k * (RS // 2) + tl.arange(0, RS // 2)
    tl.store(
        fp4_ptr + fp4_offs_m[:, None] * stride_fp4_m + fp4_offs_n[None, :],
        packed, mask=m_mask[:, None]
    )

    # Store raw scales to temp area (rows after sorted output)
    num_valid = tl.load(num_valid_ids_ptr)
    temp_row_base = m_pad

    sc_offs_m = tl.arange(0, BLOCK_M)
    sc_offs_c = tl.arange(0, NUM_QG)
    sc_m_mask = (m_base + sc_offs_m) < M
    tl.store(
        sorted_scale_ptr + (temp_row_base + m_base + sc_offs_m[:, None]) * stride_sc_m
        + (col_base + sc_offs_c[None, :]) * stride_sc_n,
        e8m0_u8, mask=sc_m_mask[:, None]
    )

    # Scatter sorted scales
    for qb in tl.static_range(0, MAX_Q // BLOCK_M):
        q = qb * BLOCK_M + tl.arange(0, BLOCK_M)
        sid = tl.load(sorted_ids_ptr + q, mask=q < num_valid, other=token_num)
        tok = sid & 0xFFFFFF
        tok_in_block = (tok >= m_base) & (tok < (m_base + BLOCK_M)) & (tok < token_num)
        q_valid_base = (q < num_valid) & tok_in_block

        cols = col_base + tl.arange(0, NUM_QG)
        col_valid = cols < N_I

        pid_n = cols >> 3
        n_local = cols & 3
        half = (cols >> 2) & 1

        q_in = q & 31
        q_half = q_in >> 4
        m_local = q_in & 15
        pid_m_out = q >> 5

        for ci in tl.static_range(0, NUM_QG):
            col_i = col_base + ci
            if col_i < N_I:
                q_valid = q_valid_base & (q < MAX_Q)
                i_val = q_half + ((col_i >> 2) & 1) * 2
                flat = (((pid_m_out * TILE_N + (col_i >> 3)) * 4
                         + (col_i & 3)) * 16 + m_local) * 4 + i_val
                dst_row = flat // N_I
                dst_col = flat % N_I

                val = tl.load(
                    sorted_scale_ptr + (temp_row_base + tok) * stride_sc_m
                    + col_i * stride_sc_n,
                    mask=q_valid, other=0
                )
                tl.store(
                    sorted_scale_ptr + dst_row * stride_sc_m + dst_col * stride_sc_n,
                    val, mask=q_valid
                )


@triton.jit
def _fused_rot_quant_sort_triton_small_m_kernel(
    x_ptr, rot_ptr, fp4_ptr,
    sorted_ids_ptr, num_valid_ids_ptr, sorted_scale_ptr,
    M, K, token_num, m_pad,
    stride_x_m, stride_x_n,
    stride_rot_r, stride_rot_c,
    stride_fp4_m,
    stride_sc_m, stride_sc_n,
    RS: tl.constexpr,
    QG: tl.constexpr,
    N_I: tl.constexpr,
    TILE_N: tl.constexpr,
    BLOCK_M_LOOP: tl.constexpr,
    m_o_padded,
):
    """M<32 specialized: each pid_k block loops over M rows with M=1-style dot product."""
    NUM_QG: tl.constexpr = RS // QG
    HALF_QG: tl.constexpr = QG // 2

    pid_k = tl.program_id(0)
    col_base = pid_k * NUM_QG

    # Load rotation once (shared across all rows)
    offs_r = tl.arange(0, RS)
    offs_c = tl.arange(0, RS)
    rot = tl.load(
        rot_ptr + offs_r[:, None] * stride_rot_r + offs_c[None, :] * stride_rot_c
    ).to(tl.float32)

    num_valid = tl.load(num_valid_ids_ptr)
    temp_row_base = m_pad

    # Process each row independently (like M=1 topk8, but in a loop)
    for row in tl.static_range(0, BLOCK_M_LOOP):
        if row < M:
            # Load x[row, RS] as 1D (same as topk8)
            offs_k = tl.arange(0, RS)
            x = tl.load(x_ptr + row * stride_x_m + (pid_k * RS + offs_k) * stride_x_n).to(tl.float32)

            # 1-row matmul (same as topk8 — fast path)
            acc = tl.dot(x[None, :].to(tl.bfloat16), rot.to(tl.bfloat16))
            acc = acc.to(tl.bfloat16).to(tl.float32)

            # Quantize
            acc_g = tl.reshape(acc, (1, NUM_QG, QG))
            amax = tl.max(tl.abs(acc_g), axis=-1, keep_dims=False)

            amax_u32 = amax.to(tl.uint32, bitcast=True)
            amax_u32 = (amax_u32 + 0x200000) & 0xFF800000
            raw_exp = (amax_u32 >> 23) & 0xFF
            e8m0 = tl.maximum(raw_exp, 2) - 2
            e8m0_u8 = e8m0.to(tl.uint8)

            hw_scale = (e8m0.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
            hw_scale_broad = tl.broadcast_to(
                tl.reshape(hw_scale, (1, NUM_QG, 1)), (1, NUM_QG, HALF_QG)
            )

            acc_pairs = tl.reshape(acc_g, (1, NUM_QG, HALF_QG, 2))
            even_vals, odd_vals = tl.split(acc_pairs)
            even_flat = tl.reshape(even_vals, (1, NUM_QG * HALF_QG))
            odd_flat = tl.reshape(odd_vals, (1, NUM_QG * HALF_QG))
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

            # Store FP4 for this row
            fp4_offs = pid_k * (RS // 2) + tl.arange(0, RS // 2)
            tl.store(fp4_ptr + row * stride_fp4_m + fp4_offs, packed)

            # Store raw scale to temp area
            sc_cols = col_base + tl.arange(0, NUM_QG)
            vals_row = tl.reshape(e8m0_u8, (NUM_QG,))
            tl.store(sorted_scale_ptr + (temp_row_base + row) * stride_sc_m + sc_cols * stride_sc_n,
                     vals_row)

    # Scatter sorted scales (process all rows' scales together)
    for qb in tl.range(0, m_o_padded, 32):
        q = qb + tl.arange(0, 32)
        sid = tl.load(sorted_ids_ptr + q, mask=q < num_valid, other=token_num)
        tok = sid & 0xFFFFFF
        q_valid_base = (q < num_valid) & (tok < token_num) & (tok < M)

        cols = col_base + tl.arange(0, NUM_QG)

        q_in = q & 31
        q_half = q_in >> 4
        m_local = q_in & 15
        pid_m_out = q >> 5

        for ci in tl.static_range(0, NUM_QG):
            col_i = col_base + ci
            if col_i < N_I:
                q_valid = q_valid_base & (q < m_o_padded)
                i_val = q_half + ((col_i >> 2) & 1) * 2
                flat = (((pid_m_out * TILE_N + (col_i >> 3)) * 4
                         + (col_i & 3)) * 16 + m_local) * 4 + i_val
                dst_row = flat // N_I
                dst_col = flat % N_I

                val = tl.load(
                    sorted_scale_ptr + (temp_row_base + tok) * stride_sc_m + col_i * stride_sc_n,
                    mask=q_valid, other=0)
                tl.store(
                    sorted_scale_ptr + dst_row * stride_sc_m + dst_col * stride_sc_n,
                    val, mask=q_valid)


def fused_rot_quant_sort_triton(
    x, rotation, rotation_size, sorted_ids, num_valid_ids, token_num, topk,
    fp4_out=None, sorted_scale_out=None,
):
    M, K = x.shape
    RS = rotation_size
    QG = 32
    n_i = K // QG
    m_o = sorted_ids.shape[0]
    m_pad = ((m_o + 31) // 32) * 32
    tile_n = triton.cdiv(n_i, 8)
    temp_rows = ((M + 31) // 32) * 32

    BLOCK_M = 32

    if fp4_out is None:
        fp4_out = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
    if sorted_scale_out is None:
        sorted_scale_out = torch.zeros((m_pad + temp_rows, n_i), dtype=torch.uint8, device=x.device)

    # Pick MAX_Q as constexpr (tight, rounded up to BLOCK_M)
    max_q = ((m_o + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    max_q = max(max_q, BLOCK_M)
    # Pick num_warps: 1 for small M (less sync), 4 for large M
    nw = 1 if M <= BLOCK_M else 4

    grid = (triton.cdiv(M, BLOCK_M), K // RS)
    _fused_rot_quant_sort_triton_kernel[grid](
        x, rotation, fp4_out,
        sorted_ids, num_valid_ids, sorted_scale_out,
        M, K, token_num, m_pad,
        x.stride(0), x.stride(1),
        rotation.stride(0), rotation.stride(1),
        fp4_out.stride(0),
        sorted_scale_out.stride(0), sorted_scale_out.stride(1),
        RS=RS, QG=QG, BLOCK_M=BLOCK_M,
        N_I=n_i, TILE_N=tile_n, MAX_Q=max_q,
        num_warps=nw,
    )
    from aiter.utility import dtypes
    return fp4_out.view(dtypes.fp4x2), sorted_scale_out[:m_pad, :n_i].view(dtypes.fp8_e8m0)
