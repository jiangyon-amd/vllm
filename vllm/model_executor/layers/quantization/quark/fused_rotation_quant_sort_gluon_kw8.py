#!/usr/bin/env python3
"""
Fused Rotation + MXFP4 Quant + MoE Sort — Gluon kw8 (3-in-1 for M>1 prefill)

Merges gluon_kw8 (rot+quant) + moe_mxfp4_sort into a single kernel.
Eliminates: 1 kernel launch (~5µs) + raw_scale HBM write+read (~8KB+).
"""

import torch
import triton
from triton import language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _fused_rot_quant_sort_kw8(
    x_ptr, rot_ptr, fp4_ptr, sorted_scale_ptr,
    sorted_ids_ptr, num_valid_ids_ptr,
    M, K, token_num, m_pad,
    stride_x_m, stride_rot_r, stride_rot_c,
    stride_fp4_m, stride_sc_m, stride_sc_n,
    RS: gl.constexpr,
    QG: gl.constexpr,
    BLOCK_M: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    N_I: gl.constexpr,
    TILE_N: gl.constexpr,
    MAX_Q: gl.constexpr,
):
    pid_m   = gl.program_id(0)
    pid_rot = gl.program_id(1)

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
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=8)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=8)

    NUM_QG: gl.constexpr = RS // QG
    HALF_QG: gl.constexpr = QG // 2

    m_base = pid_m * BLOCK_M
    col_base = pid_rot * RS
    offs_m = m_base + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked_mk))
    m_mask = offs_m < M

    # ======== Load x + rotation + MFMA matmul (same as gluon_kw8) ========
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
    acc = acc.to(gl.bfloat16).to(gl.float32)

    # ======== MXFP4 quantization (same as gluon_kw8) ========
    acc_g = gl.reshape(acc, (BLOCK_M, NUM_QG, QG))
    amax = gl.max(gl.abs(acc_g), axis=-1)

    amax_u32 = amax.to(gl.uint32, bitcast=True)
    amax_u32 = (amax_u32 + 0x200000) & 0xFF800000
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

    # ======== Store FP4 (same as gluon_kw8) ========
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

    # ======== FUSED: Scatter with temp rows at end of buffer ========
    sc_store2: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1], threads_per_warp=[16, 4],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0],
    )
    sc_s = gl.convert_layout(e8m0_u8, sc_store2)
    sc_tmp_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, sc_store2))
    sc_tmp_mask = sc_tmp_m < M - m_base
    sc_tmp_c = gl.arange(0, NUM_QG, layout=gl.SliceLayout(0, sc_store2))
    col = pid_rot * NUM_QG + sc_tmp_c

    num_valid = tl.load(num_valid_ids_ptr)
    temp_row_base = m_pad

    # Write raw scales to temp area (non-overlapping with sorted output)
    tl.store(sorted_scale_ptr + (temp_row_base + m_base + sc_tmp_m[:, None]) * stride_sc_m
             + col[None, :] * stride_sc_n,
             sc_s, mask=sc_tmp_mask[:, None])

    # Scatter from temp area to sorted layout
    sc_scatter: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1], threads_per_warp=[16, 4],
        warps_per_cta=[4, 1], order=[1, 0],
    )

    for qb in tl.static_range(0, MAX_Q // 64):
        q = qb * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, sc_scatter))
        c = gl.arange(0, NUM_QG, layout=gl.SliceLayout(0, sc_scatter))
        qcol = pid_rot * NUM_QG + c

        sid = tl.load(sorted_ids_ptr + q[:, None] + c[None, :] * 0,
                       mask=q[:, None] < num_valid, other=token_num)
        tok = sid & 0xFFFFFF
        tok_in_block = (tok >= m_base) & (tok < (m_base + BLOCK_M)) & (tok < token_num)
        q_valid = (q[:, None] < num_valid) & tok_in_block

        pid_n = qcol >> 3
        n_local = qcol & 3
        half = (qcol >> 2) & 1
        q_in = q & 31
        q_half = q_in >> 4
        m_local = q_in & 15
        pid_m_out = q >> 5
        i = q_half[:, None] + half[None, :] * 2

        flat = (((pid_m_out[:, None] * TILE_N + pid_n[None, :]) * 4
                 + n_local[None, :]) * 16 + m_local[:, None]) * 4 + i
        dst_row = flat // N_I
        dst_col = flat % N_I

        val = tl.load(sorted_scale_ptr + (temp_row_base + tok) * stride_sc_m
                       + qcol[None, :] * stride_sc_n,
                       mask=q_valid, other=0)
        tl.store(
            sorted_scale_ptr + dst_row * stride_sc_m + dst_col * stride_sc_n,
            val, mask=q_valid,
        )


def _pick_max_q(m_o: int) -> int:
    """Pick MAX_Q as tight power-of-2-multiple-of-64 >= m_o."""
    mq = ((m_o + 63) // 64) * 64
    return max(mq, 64)


def fused_rot_quant_sort_kw8(
    x, rotation, rotation_size, sorted_ids, num_valid_ids, token_num, topk,
    block_size=32,
    fp4_out=None, sorted_scale_out=None,
):
    M, K = x.shape
    RS = rotation_size
    QG = 32
    n_i = K // QG
    m_o = sorted_ids.shape[0]
    m_pad = ((m_o + 31) // 32) * 32
    tile_n = triton.cdiv(n_i, 8)
    max_q = _pick_max_q(m_o)
    temp_rows = ((M + 31) // 32) * 32

    if fp4_out is None:
        fp4_out = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
    if sorted_scale_out is None:
        sorted_scale_out = torch.zeros((m_pad + temp_rows, n_i), dtype=torch.uint8, device=x.device)

    BLOCK_M = 32
    NUM_WARPS = 4
    grid = (triton.cdiv(M, BLOCK_M), K // RS)
    _fused_rot_quant_sort_kw8[grid](
        x, rotation, fp4_out, sorted_scale_out,
        sorted_ids, num_valid_ids,
        M, K, token_num, m_pad,
        x.stride(0), rotation.stride(0), rotation.stride(1),
        fp4_out.stride(0), sorted_scale_out.stride(0), sorted_scale_out.stride(1),
        RS=RS, QG=QG, BLOCK_M=BLOCK_M,
        NUM_WARPS=NUM_WARPS, num_warps=NUM_WARPS,
        N_I=n_i, TILE_N=tile_n, MAX_Q=max_q,
    )
    from aiter.utility import dtypes
    return fp4_out.view(dtypes.fp4x2), sorted_scale_out[:m_pad, :n_i].view(dtypes.fp8_e8m0)
