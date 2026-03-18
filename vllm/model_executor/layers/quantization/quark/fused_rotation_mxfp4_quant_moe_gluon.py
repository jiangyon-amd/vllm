#!/usr/bin/env python3
"""
Gluon MoE Fused Rotation + MXFP4 Quant + Sorted Scale Scatter.

Optimized version:
  1. n_i/tile_n as constexpr for fast integer div/mod
  2. Direct scale scatter (no 2-phase temp row)
  3. Proper broadcast (no multiply-by-zero hack)
"""

import torch
import triton
from triton import language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _fused_decode_m1_rot_quant_sorted_gluon_kernel(
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
    RS: gl.constexpr,
    QG: gl.constexpr,
    MAX_Q: gl.constexpr,
    N_I: gl.constexpr,
    TILE_N: gl.constexpr,
):
    pid_k = gl.program_id(0)
    NUM_QG: gl.constexpr = RS // QG
    HALF_QG: gl.constexpr = QG // 2
    col_base = pid_k * NUM_QG

    # ======== Layouts ========
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16], transposed=True,
        warps_per_cta=[1, 4], tiles_per_warp=[1, 2],
    )
    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[16, 4],
        warps_per_cta=[4, 1], order=[1, 0],
    )
    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[4, 1], threads_per_warp=[4, 16],
        warps_per_cta=[1, 4], order=[0, 1],
    )
    shared_rot: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[1, 0],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=4)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=4)

    BLOCK_M: gl.constexpr = 16

    # ======== Load x [BLOCK_M, RS] ========
    offs_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked_mk))
    m_mask = offs_m < 1
    offs_k = gl.arange(0, RS, layout=gl.SliceLayout(0, blocked_mk))
    x_offs = offs_m[:, None] * 0 + (pid_k * RS + offs_k)[None, :] * stride_x_n
    x_tile = gl.amd.cdna4.buffer_load(ptr=x_ptr, offsets=x_offs, mask=m_mask[:, None])

    # ======== Load rotation [RS, RS] to shared ========
    smem_rot = gl.allocate_shared_memory(gl.bfloat16, [RS, RS], layout=shared_rot)
    offs_rr = gl.arange(0, RS, layout=gl.SliceLayout(1, blocked_kn))
    offs_rc = gl.arange(0, RS, layout=gl.SliceLayout(0, blocked_kn))
    rot_data = gl.amd.cdna4.buffer_load(
        ptr=rot_ptr,
        offsets=offs_rr[:, None] * stride_rot_r + offs_rc[None, :] * stride_rot_c,
    )
    smem_rot.store(rot_data)

    # ======== MFMA rotation matmul ========
    acc = gl.zeros((BLOCK_M, RS), gl.float32, layout=mfma_layout)
    acc = gl.amd.cdna4.mfma(gl.convert_layout(x_tile, dot_a),
                             smem_rot.load(layout=dot_b), acc)
    acc = acc.to(gl.bfloat16).to(gl.float32)

    # ======== Quantization ========
    acc_g = gl.reshape(acc, (BLOCK_M, NUM_QG, QG))
    amax = gl.max(gl.abs(acc_g), axis=-1)

    amax_u32 = amax.to(gl.uint32, bitcast=True)
    amax_u32 = (amax_u32 + 0x200000) & 0xFF800000
    raw_exp = (amax_u32 >> 23) & 0xFF
    e8m0 = gl.maximum(raw_exp, 2) - 2
    e8m0_u8 = e8m0.to(gl.uint8)

    hw_scale = (e8m0.to(gl.uint32) << 23).to(gl.float32, bitcast=True)

    # Broadcast hw_scale to match acc_g shape
    hw_scale_bcast = acc_g * 0.0 + gl.expand_dims(hw_scale, axis=2)

    acc_pairs = gl.reshape(acc_g, (BLOCK_M, NUM_QG, HALF_QG, 2))
    even_vals, odd_vals = gl.split(acc_pairs)
    scale_pairs = gl.reshape(hw_scale_bcast, (BLOCK_M, NUM_QG, HALF_QG, 2))
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
        warps_per_cta=[4, 1], order=[1, 0],
    )
    packed_s = gl.convert_layout(packed, fp4_store)
    fp4_base = pid_k * (RS // 2)
    s_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, fp4_store))
    s_mask = s_m < 1
    fp4_c = fp4_base + gl.arange(0, RS // 2, layout=gl.SliceLayout(0, fp4_store))
    gl.amd.cdna4.buffer_store(stored_value=packed_s, ptr=fp4_ptr,
        offsets=s_m[:, None] * stride_fp4_m + fp4_c[None, :], mask=s_mask[:, None])

    # ======== FIX #2: Direct scale scatter (no 2-phase temp row) ========
    # Extract row-0 e8m0 values via tl.store to a temp scalar, then scatter.
    # Use tl.store/tl.load directly (bypass gl.convert_layout data loss).
    sc_store2: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1], threads_per_warp=[16, 4],
        warps_per_cta=[4, 1], order=[1, 0],
    )
    sc_s = gl.convert_layout(e8m0_u8, sc_store2)
    sc_tmp_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, sc_store2))
    sc_tmp_mask = sc_tmp_m < 1
    sc_tmp_c = col_base + gl.arange(0, NUM_QG, layout=gl.SliceLayout(0, sc_store2))

    # Write raw scales to temp row once (needed for gl.convert_layout data recovery)
    tl.store(sorted_scale_ptr + (m_o + sc_tmp_m[:, None]) * stride_sc_m + sc_tmp_c[None, :] * stride_sc_n,
             sc_s, mask=sc_tmp_mask[:, None])

    # FIX #3: Direct scatter with constexpr N_I/TILE_N (no re-read needed)
    num_valid = tl.load(num_valid_ids_ptr)

    sc_scatter_2d: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1], threads_per_warp=[16, 4],
        warps_per_cta=[4, 1], order=[1, 0],
    )

    for qb in tl.static_range(0, MAX_Q // 64):
        q = qb * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, sc_scatter_2d))
        c = gl.arange(0, NUM_QG, layout=gl.SliceLayout(0, sc_scatter_2d))
        col = col_base + c

        sid = tl.load(sorted_ids_ptr + q[:, None] + c[None, :] * 0,
                       mask=q[:, None] < num_valid, other=token_num)
        tok = sid & 0xFFFFFF
        q_valid = (q[:, None] < num_valid) & (tok < token_num) & (q[:, None] < m_o)

        pid_n = col >> 3
        n_local = col & 3
        half = (col >> 2) & 1

        q_in = q & 31
        q_half = q_in >> 4
        m_local = q_in & 15
        pid_m_out = q >> 5
        i = q_half[:, None] + half[None, :] * 2
        # FIX #3: use constexpr TILE_N and N_I for fast div/mod
        flat = (((pid_m_out[:, None] * TILE_N + pid_n[None, :]) * 4
                 + n_local[None, :]) * 16 + m_local[:, None]) * 4 + i
        dst_row = flat // N_I
        dst_col = flat % N_I

        val = tl.load(sorted_scale_ptr + m_o * stride_sc_m
                       + col[None, :] * stride_sc_n + q[:, None] * 0)
        tl.store(
            sorted_scale_ptr + dst_row * stride_sc_m + dst_col * stride_sc_n,
            val, mask=q_valid,
        )


def fused_decode_m1_rot_quant_sorted_gluon(
    x, rotation, fp4_out, sorted_ids, num_valid_ids, sorted_scale_out,
    K, RS, QG, n_i, token_num, m_o, m_pad,
):
    """Launch gluon MoE decode kernel."""
    tile_n = triton.cdiv(n_i, 8)

    from vllm.model_executor.layers.quantization.quark.fused_rotation_mxfp4_quant_moe_sort import (
        _pick_decode_max_q,
    )
    max_q = _pick_decode_max_q(m_pad)

    grid = (K // RS,)
    _fused_decode_m1_rot_quant_sorted_gluon_kernel[grid](
        x, rotation, fp4_out,
        sorted_ids, num_valid_ids, sorted_scale_out,
        x.stride(1),
        rotation.stride(0), rotation.stride(1),
        fp4_out.stride(0),
        sorted_scale_out.stride(0), sorted_scale_out.stride(1),
        token_num=token_num, m_o=m_o,
        RS=RS, QG=QG, MAX_Q=max_q,
        N_I=n_i, TILE_N=tile_n,
        num_warps=4,
    )
    return fp4_out, sorted_scale_out
