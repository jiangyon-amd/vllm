#!/usr/bin/env python3
"""
Fused Rotation + MXFP4 Quant — Gluon v2

Single kernel: rotation matmul (MFMA) + MXFP4 quantization + scale shuffle.
For dense models (Qwen3-8B/14B/32B). MoE paths are in separate files.
"""

import torch
import triton
from triton import language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

if not (hasattr(gl, 'amd') and hasattr(gl.amd, 'cdna4')):
    raise ImportError("Gluon cdna4 not available (requires Triton 3.5.x)")


@gluon.jit
def _fused_rot_quant_v2(
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

    # ======== Load x [BLOCK_M, RS] via buffer_load ========
    offs_k = gl.arange(0, RS, layout=gl.SliceLayout(0, blocked_mk))
    x_offs = offs_m[:, None] * stride_x_m + (col_base + offs_k)[None, :]
    x_tile = gl.amd.cdna4.buffer_load(ptr=x_ptr, offsets=x_offs, mask=m_mask[:, None])

    # ======== Load rotation [RS, RS] to shared memory ========
    smem_rot = gl.allocate_shared_memory(gl.bfloat16, [RS, RS], layout=shared_rot)
    offs_rr = gl.arange(0, RS, layout=gl.SliceLayout(1, blocked_kn))
    offs_rc = gl.arange(0, RS, layout=gl.SliceLayout(0, blocked_kn))
    rot_data = gl.amd.cdna4.buffer_load(ptr=rot_ptr,
        offsets=offs_rr[:, None] * stride_rot_r + offs_rc[None, :] * stride_rot_c)
    smem_rot.store(rot_data)

    # ======== MFMA rotation matmul (single pass, no K accumulation) ========
    acc = gl.zeros((BLOCK_M, RS), gl.float32, layout=mfma_layout)
    acc = gl.amd.cdna4.mfma(gl.convert_layout(x_tile, dot_a),
                             smem_rot.load(layout=dot_b), acc)
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

    # ======== Store FP4 via buffer_store ========
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
    sc_store2: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1], threads_per_warp=[32, 2],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0],
    )
    sc_s = gl.convert_layout(e8m0_u8, sc_store2)
    sc_m_idx = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, sc_store2))
    sc_q_idx = gl.arange(0, NUM_QG, layout=gl.SliceLayout(0, sc_store2))
    sc_mask = (m_base + sc_m_idx[:, None]) < M
    sc_base = pid_rot * NUM_QG

    if SHUFFLE_SCALES:
        # Two-phase shuffle via global temp rows.
        # gl.convert_layout loses data for NUM_QG>2, but raw store with
        # SliceLayout broadcasting writes all values correctly.
        raw_off = ((M + 31) // 32) * 32
        # Phase 1: raw store to temp rows
        tl.store(scale_ptr + (m_base + sc_m_idx[:, None] + raw_off) * stride_sc_m
                 + (sc_base + sc_q_idx[None, :]),
                 sc_s, mask=sc_mask)
        # Phase 2: load from temp, scatter to shuffled positions
        raw_val = tl.load(scale_ptr + (m_base + sc_m_idx[:, None] + raw_off) * stride_sc_m
                          + (sc_base + sc_q_idx[None, :]),
                          mask=sc_mask, other=0)
        col = sc_base + sc_q_idx
        flat = ((col >> 3)[None, :] * 256
                + (col & 3)[None, :] * 64
                + (sc_m_idx & 15)[:, None] * 4
                + ((col >> 2) & 1)[None, :] * 2
                + (sc_m_idx >> 4)[:, None])
        tl.store(scale_ptr + (flat // sn_padded + m_base) * stride_sc_m + flat % sn_padded,
                 raw_val, mask=sc_mask)
    else:
        sc_m = m_base + sc_m_idx
        sc_c = sc_base + sc_q_idx
        tl.store(scale_ptr + sc_m[:, None] * stride_sc_m + sc_c[None, :],
                 sc_s, mask=sc_mask)


def fused_gluon_v2(
    x, rotation, rotation_size=128,
    fp4_out=None, scales_out=None, shuffle_scales=False,
    # MoE params (ignored, kept for API compatibility)
    sorted_scales=False, sorted_scales_topk8=False,
    sorted_ids=None, num_valid_ids=None, token_num=0,
):
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
        raw_row_offset = ((M + 31) // 32) * 32
        sm_with_temp = max(sm_padded, raw_row_offset + ((M + 31) // 32) * 32)
        if scales_out is None:
            scales_out = torch.empty((sm_with_temp, sn_padded), dtype=torch.uint8, device=x.device)
        elif scales_out.shape[0] < sm_with_temp:
            scales_out = torch.empty((sm_with_temp, sn_padded), dtype=torch.uint8, device=x.device)
    else:
        sn_padded = n_scales
        if scales_out is None:
            scales_out = torch.empty((M, n_scales), dtype=torch.uint8, device=x.device)

    BLOCK_M = 32
    NUM_WARPS = 4
    grid = (triton.cdiv(M, BLOCK_M), K // RS)
    _fused_rot_quant_v2[grid](
        x, rotation, fp4_out, scales_out, M, K, sn_padded,
        x.stride(0), rotation.stride(0), rotation.stride(1),
        fp4_out.stride(0), scales_out.stride(0),
        RS=RS, QG=QGROUP, BLOCK_M=BLOCK_M,
        NUM_WARPS=NUM_WARPS, num_warps=NUM_WARPS,
        SHUFFLE_SCALES=shuffle_scales,
    )
    return fp4_out, scales_out
