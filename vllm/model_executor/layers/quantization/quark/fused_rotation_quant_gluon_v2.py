#!/usr/bin/env python3
"""
Fused Rotation + MXFP4 Quant — Gluon v2

Key change from v16: use tl.store for scale scatter (no convert_layout needed).
Keep Gluon's MFMA + buffer_load/buffer_store for x/rotation/fp4,
but use tl.store for shuffle scale store to avoid convert_layout overhead.
"""

import torch
import triton
from triton import language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import logging
import os
from aiter.utility.fp4_utils import moe_mxfp4_sort

logger = logging.getLogger(__name__)
ENABLE_FUSED_MOE_DEBUG_LOG = (
    os.getenv("VLLM_MOE_FUSED_DEBUG_LOG", "0") == "1"
)
ENABLE_GLUON_TOPK8_FALLBACK = (
    os.getenv("VLLM_MOE_GLUON_TOPK8_FALLBACK", "0") == "1"
)
_GLUON_LAUNCH_LOG_KEYS: set[tuple] = set()


def _pick_decode_max_q(m_pad: int) -> int:
    return 256


@gluon.jit
def _fused_rot_quant_decode_topk8_special(
    x_ptr,
    rot_ptr,
    fp4_ptr,
    scale_ptr,
    sorted_ids_ptr,
    num_valid_ids_ptr,
    K,
    stride_x_m,
    stride_rot_r,
    stride_rot_c,
    stride_fp4_m,
    stride_sc_m,
    stride_sid,
    token_num,
    m_o,
    RS: gl.constexpr,
    QG: gl.constexpr,
    MAX_Q: gl.constexpr,
):
    pid_rot = gl.program_id(1)
    NUM_QG: gl.constexpr = RS // QG
    HALF_QG: gl.constexpr = QG // 2
    col_base = pid_rot * RS

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16], transposed=True,
        warps_per_cta=[2, 2], tiles_per_warp=[1, 4],
    )
    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4], threads_per_warp=[16, 4],
        warps_per_cta=[4, 1], order=[1, 0],
    )
    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[4, 1], threads_per_warp=[4, 16],
        warps_per_cta=[1, 4], order=[0, 1],
    )
    shared_rot: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[1, 0],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=8)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=8)

    BLOCK_M: gl.constexpr = 32
    offs_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked_mk))
    m_mask = offs_m < 1
    offs_k = gl.arange(0, RS, layout=gl.SliceLayout(0, blocked_mk))
    x_offs = offs_m[:, None] * stride_x_m + (col_base + offs_k)[None, :]
    x_tile = gl.amd.cdna4.buffer_load(ptr=x_ptr, offsets=x_offs, mask=m_mask[:, None])

    smem_rot = gl.allocate_shared_memory(gl.bfloat16, [RS, RS], layout=shared_rot)
    offs_rr = gl.arange(0, RS, layout=gl.SliceLayout(1, blocked_kn))
    offs_rc = gl.arange(0, RS, layout=gl.SliceLayout(0, blocked_kn))
    rot_data = gl.amd.cdna4.buffer_load(
        ptr=rot_ptr,
        offsets=offs_rr[:, None] * stride_rot_r + offs_rc[None, :] * stride_rot_c,
    )
    smem_rot.store(rot_data)

    acc = gl.zeros((BLOCK_M, RS), gl.float32, layout=mfma_layout)
    acc = gl.amd.cdna4.mfma(gl.convert_layout(x_tile, dot_a), smem_rot.load(layout=dot_b), acc)
    acc = acc.to(gl.bfloat16).to(gl.float32)

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

    fp4_store: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4], threads_per_warp=[2, 32],
        warps_per_cta=[4, 1], order=[1, 0],
    )
    packed_s = gl.convert_layout(packed, fp4_store)
    fp4_base = pid_rot * (RS // 2)
    s_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, fp4_store))
    s_mask = s_m < 1
    fp4_c = fp4_base + gl.arange(0, RS // 2, layout=gl.SliceLayout(0, fp4_store))
    tl.store(fp4_ptr + s_m[:, None] * stride_fp4_m + fp4_c[None, :], packed_s, mask=s_mask[:, None])

    # Write raw scale to row-0 of scale_ptr, then do scatter in a second phase.
    sc_store2: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1], threads_per_warp=[32, 2],
        warps_per_cta=[4, 1], order=[1, 0],
    )
    sc_s = gl.convert_layout(e8m0_u8, sc_store2)
    # Write raw scales to a temp row beyond the valid output area (row = m_o).
    sc_tmp_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, sc_store2))
    sc_tmp_mask = sc_tmp_m < 1
    sc_tmp_c = pid_rot * NUM_QG + gl.arange(0, NUM_QG, layout=gl.SliceLayout(0, sc_store2))
    tl.store(scale_ptr + (m_o + sc_tmp_m[:, None]) * stride_sc_m + sc_tmp_c[None, :], sc_s, mask=sc_tmp_mask[:, None])

    # Scatter sorted scale: loop over NUM_QG columns (outer), then q-blocks (inner).
    # This way each scale value is loaded once from the temp row and reused.
    sc_scatter: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1], threads_per_warp=[64, 1],
        warps_per_cta=[4, 1], order=[1, 0],
    )
    num_valid = tl.load(num_valid_ids_ptr)
    n_i = K // QG
    tile_n = (n_i + 7) // 8

    for c in tl.static_range(0, NUM_QG):
        col = pid_rot * NUM_QG + c
        pid_n = col >> 3
        n_local = col & 3
        half = (col & 7) >> 2

        for qb in tl.static_range(0, MAX_Q // 64):
            q = qb * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, sc_scatter))
            sid = tl.load(sorted_ids_ptr + q, mask=q < num_valid, other=token_num)
            tok = sid & 0xFFFFFF
            q_valid = (q < num_valid) & (tok < token_num) & (q < m_o)

            q_in = q & 31
            q_half = q_in >> 4
            m_local = q_in & 15
            pid_m_out = q >> 5
            i = q_half + half * 2
            flat = (((pid_m_out * tile_n + pid_n) * 4 + n_local) * 16 + m_local) * 4 + i
            dst_row = flat // n_i
            dst_col = flat % n_i
            val = tl.load(scale_ptr + m_o * stride_sc_m + col + q * 0)
            tl.store(
                scale_ptr + dst_row * stride_sc_m + dst_col,
                val,
                mask=q_valid,
            )


@gluon.jit
def _fused_rot_quant_v2(
    x_ptr, rot_ptr, fp4_ptr, scale_ptr,
    sorted_ids_ptr, num_valid_ids_ptr,
    M, K, sn_padded,
    stride_x_m, stride_rot_r, stride_rot_c,
    stride_fp4_m, stride_sc_m,
    stride_sid,
    token_num, m_o,
    RS: gl.constexpr,
    QG: gl.constexpr,
    BLOCK_M: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SHUFFLE_SCALES: gl.constexpr,
    SORTED_SCALES: gl.constexpr,
    SORTED_SCALES_TOPK8: gl.constexpr,
    MAX_Q: gl.constexpr,
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

    # ======== MFMA rotation matmul ========
    acc = gl.zeros((BLOCK_M, RS), gl.float32, layout=mfma_layout)
    acc = gl.amd.cdna4.mfma(gl.convert_layout(x_tile, dot_a),
                             smem_rot.load(layout=dot_b), acc)
    # Truncate to bf16 precision to match torch matmul
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
    # Use fp4_store layout for scale convert_layout (reuse fp4 store's layout
    # which is already computed, avoiding a separate convert_layout)
    # fp4_store covers [BLOCK_M, RS//2] with [1,4] per thread
    # scales are [BLOCK_M, NUM_QG] — much smaller, but same M dimension
    sc_store: gl.constexpr = fp4_store  # Reuse fp4_store layout!

    if SORTED_SCALES_TOPK8:
        # Decode specialized path (token_num=1, topk=8):
        # use row-0 scales and simplified index math.
        if pid_m == 0:
            num_valid = tl.load(num_valid_ids_ptr)
            lane = gl.arange(0, 32, layout=gl.SliceLayout(1, fp4_store))[:, None]
            vals = tl.broadcast_to(e8m0_u8, (32, NUM_QG))
            q_half = lane // 16
            m_local = lane % 16
            n_local = gl.arange(0, NUM_QG, layout=gl.SliceLayout(0, fp4_store))[None, :]
            pid_n = pid_rot // 2
            half = pid_rot % 2
            i = q_half + 2 * half
            tile_n = sn_padded // 8

            for qb in tl.static_range(0, MAX_Q // 32):
                q = (qb * 32 + lane)
                sid = tl.load(sorted_ids_ptr + q * stride_sid, mask=q < num_valid, other=token_num)
                tok = sid & 0xFFFFFF
                q_valid = (q < num_valid) & (tok < token_num) & (q < m_o)

                flat = ((((qb * tile_n + pid_n) * 4 + n_local) * 16 + m_local) * 4 + i)
                dst_row = flat // sn_padded
                dst_col = flat % sn_padded
                tl.store(scale_ptr + dst_row * stride_sc_m + dst_col, vals, mask=q_valid)
    elif SORTED_SCALES:
        # Decode sorted-scale store: compute rot/quant once per pid_rot,
        # then iterate q-blocks for scatter write.
        if pid_m == 0:
            num_valid = tl.load(num_valid_ids_ptr)
            col = pid_rot * NUM_QG + tl.arange(0, NUM_QG)[None, :]
            col_valid = col < sn_padded
            pid_n = col // 8
            n_local = col % 4
            half = (col % 8) // 4
            tile_n = (sn_padded + 7) // 8
            lane = gl.arange(0, 32, layout=gl.SliceLayout(1, fp4_store))[:, None]
            vals = tl.broadcast_to(e8m0_u8, (32, NUM_QG))
            q_half = lane // 16
            m_local = lane % 16
            for qb in tl.static_range(0, MAX_Q // 32):
                q = qb * 32 + lane
                sid = tl.load(sorted_ids_ptr + q * stride_sid, mask=q < num_valid, other=token_num)
                tok = sid & 0xFFFFFF
                q_valid = (q < num_valid) & (tok < token_num)
                i = q_half + 2 * half
                pid_m_out = qb + 0 * lane
                flat = ((((pid_m_out * tile_n + pid_n) * 4 + n_local) * 16 + m_local) * 4 + i)
                dst_row = flat // sn_padded
                dst_col = flat % sn_padded
                mask = q_valid & col_valid
                tl.store(scale_ptr + dst_row * stride_sc_m + dst_col, vals, mask=mask)
    elif SHUFFLE_SCALES:
        # Pad e8m0 to match fp4_store's N dimension, then extract
        # Actually we can't reuse fp4_store directly (shape mismatch)
        # Use a minimal blocked layout instead
        sc_store2: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1], threads_per_warp=[32, 2],
            warps_per_cta=[NUM_WARPS, 1], order=[1, 0],
        )
        sc_s = gl.convert_layout(e8m0_u8, sc_store2)

        sc_m_local = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, sc_store2))
        sc_q = gl.arange(0, NUM_QG, layout=gl.SliceLayout(0, sc_store2))
        orig_col = pid_rot * NUM_QG + sc_q
        flat_idx = ((orig_col // 8)[None, :] * 256
                  + (orig_col % 4)[None, :] * 64
                  + (sc_m_local % 16)[:, None] * 4
                  + ((orig_col % 8) // 4)[None, :] * 2
                  + (sc_m_local // 16)[:, None])
        sh_row = flat_idx // sn_padded + m_base
        sh_col = flat_idx % sn_padded
        sc_mask_sh = sh_row < (m_base + BLOCK_M)
        tl.store(scale_ptr + sh_row * stride_sc_m + sh_col, sc_s, mask=sc_mask_sh)
    else:
        sc_store2: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1], threads_per_warp=[32, 2],
            warps_per_cta=[NUM_WARPS, 1], order=[1, 0],
        )
        sc_s = gl.convert_layout(e8m0_u8, sc_store2)
        sc_m = m_base + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, sc_store2))
        sc_mask = sc_m < M
        sc_base = pid_rot * NUM_QG
        sc_c = sc_base + gl.arange(0, NUM_QG, layout=gl.SliceLayout(0, sc_store2))
        tl.store(scale_ptr + sc_m[:, None] * stride_sc_m + sc_c[None, :], sc_s, mask=sc_mask[:, None])


def fused_gluon_v2(
    x,
    rotation,
    rotation_size=128,
    fp4_out=None,
    scales_out=None,
    shuffle_scales=False,
    sorted_scales=False,
    sorted_scales_topk8=False,
    sorted_ids=None,
    num_valid_ids=None,
    token_num=0,
):
    assert x.ndim == 2
    M, K = x.shape
    assert K % rotation_size == 0
    RS = rotation_size
    QGROUP = 32
    n_scales = K // QGROUP

    if fp4_out is None:
        fp4_out = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)

    # Optional compatibility fallback for decode topk8 sorted-scales path:
    # run Gluon rot+quant to raw scales, then reuse moe_mxfp4_sort.
    if sorted_scales_topk8 and ENABLE_GLUON_TOPK8_FALLBACK:
        assert sorted_ids is not None and num_valid_ids is not None
        assert x.shape[0] == 1 and token_num == 1
        raw_scale = torch.empty((M, n_scales), dtype=torch.uint8, device=x.device)
        sn_padded = (n_scales + 7) // 8 * 8
        sm_padded = ((sorted_ids.shape[0] + 31) // 32) * 32
        if scales_out is None:
            scales_out = torch.empty((sm_padded, sn_padded), dtype=torch.uint8, device=x.device)

        BLOCK_M = 16 if M <= 16 else 32
        NUM_WARPS = 4
        NUM_STAGES = 2 if M <= 16 else 3
        grid = (triton.cdiv(M, BLOCK_M), K // RS)
        _fused_rot_quant_v2[grid](
            x,
            rotation,
            fp4_out,
            raw_scale,
            x,
            x,
            M,
            K,
            n_scales,
            x.stride(0),
            rotation.stride(0),
            rotation.stride(1),
            fp4_out.stride(0),
            raw_scale.stride(0),
            1,
            0,
            0,
            RS=RS,
            QG=QGROUP,
            BLOCK_M=BLOCK_M,
            NUM_WARPS=NUM_WARPS,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            SHUFFLE_SCALES=False,
            SORTED_SCALES=False,
            SORTED_SCALES_TOPK8=False,
            MAX_Q=256,
        )
        sorted_scale = moe_mxfp4_sort(
            raw_scale,
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=token_num,
            block_size=32,
        ).view(torch.uint8)
        if scales_out.shape == sorted_scale.shape:
            scales_out.copy_(sorted_scale)
        else:
            scales_out.zero_()
            scales_out[:, :n_scales].copy_(sorted_scale)
        return fp4_out, scales_out

    if sorted_scales_topk8:
        assert sorted_ids is not None and num_valid_ids is not None
        assert x.shape[0] == 1 and token_num == 1
        sn_padded = (n_scales + 7) // 8 * 8
        sm_padded = ((sorted_ids.shape[0] + 31) // 32) * 32
        max_q = _pick_decode_max_q(sm_padded)
        if scales_out is None:
            scales_out = torch.empty((sm_padded + 32, sn_padded), dtype=torch.uint8, device=x.device)
        elif scales_out.shape[0] < sm_padded + 32:
            scales_out = torch.empty((sm_padded + 32, sn_padded), dtype=torch.uint8, device=x.device)
        grid = (1, K // RS)
        _fused_rot_quant_decode_topk8_special[grid](
            x,
            rotation,
            fp4_out,
            scales_out,
            sorted_ids,
            num_valid_ids,
            K,
            x.stride(0),
            rotation.stride(0),
            rotation.stride(1),
            fp4_out.stride(0),
            scales_out.stride(0),
            sorted_ids.stride(0),
            token_num,
            sorted_ids.shape[0],
            RS=RS,
            QG=QGROUP,
            MAX_Q=max_q,
            num_warps=4,
            num_stages=2,
        )
        return fp4_out, scales_out
    elif sorted_scales:
        assert sorted_ids is not None and num_valid_ids is not None
        assert x.shape[0] == 1, "sorted_scales path currently supports decode (M=1)"
        sn_padded = (n_scales + 7) // 8 * 8
        sm_padded = ((sorted_ids.shape[0] + 31) // 32) * 32
        max_q = _pick_decode_max_q(sm_padded)
        if scales_out is None:
            scales_out = torch.empty((sm_padded, sn_padded), dtype=torch.uint8, device=x.device)
    elif shuffle_scales:
        sn_padded = (n_scales + 7) // 8 * 8
        sm_padded = (M + 255) // 256 * 256
        max_q = 256
        if scales_out is None:
            scales_out = torch.empty((sm_padded, sn_padded), dtype=torch.uint8, device=x.device)
    else:
        sn_padded = n_scales
        max_q = 256
        if scales_out is None:
            scales_out = torch.empty((M, n_scales), dtype=torch.uint8, device=x.device)

    # Conservative adaptive launch config inspired by HIP fused path:
    # - keep sorted-scale paths on BLOCK_M=32 (kernel indexing assumes 32 lanes)
    # - optimize regular fused path for small M with lighter CTAs
    if sorted_scales or sorted_scales_topk8:
        BLOCK_M = 32
        NUM_WARPS = 4
        NUM_STAGES = 3
    elif M <= 16:
        BLOCK_M = 16
        NUM_WARPS = 4
        NUM_STAGES = 2
    else:
        BLOCK_M = 32
        NUM_WARPS = 4
        NUM_STAGES = 3
    if ENABLE_FUSED_MOE_DEBUG_LOG:
        key = (
            M,
            K,
            RS,
            shuffle_scales,
            sorted_scales,
            sorted_scales_topk8,
            BLOCK_M,
            NUM_WARPS,
            NUM_STAGES,
        )
        if key not in _GLUON_LAUNCH_LOG_KEYS:
            _GLUON_LAUNCH_LOG_KEYS.add(key)
            logger.info(
                "fused_gluon_v2 launch: M=%s K=%s RS=%s shuffle=%s sorted=%s sorted_topk8=%s "
                "BLOCK_M=%s NUM_WARPS=%s NUM_STAGES=%s",
                M,
                K,
                RS,
                shuffle_scales,
                sorted_scales,
                sorted_scales_topk8,
                BLOCK_M,
                NUM_WARPS,
                NUM_STAGES,
            )
    grid = (triton.cdiv(M, BLOCK_M), K // RS)
    _fused_rot_quant_v2[grid](
        x, rotation, fp4_out, scales_out,
        sorted_ids if sorted_ids is not None else x,
        num_valid_ids if num_valid_ids is not None else x,
        M, K, sn_padded,
        x.stride(0), rotation.stride(0), rotation.stride(1),
        fp4_out.stride(0), scales_out.stride(0),
        sorted_ids.stride(0) if sorted_ids is not None else 1,
        token_num, sorted_ids.shape[0] if sorted_ids is not None else 0,
        RS=RS, QG=QGROUP, BLOCK_M=BLOCK_M,
        NUM_WARPS=NUM_WARPS, num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        SHUFFLE_SCALES=shuffle_scales,
        SORTED_SCALES=sorted_scales,
        SORTED_SCALES_TOPK8=sorted_scales_topk8,
        MAX_Q=max_q,
    )
    return fp4_out, scales_out
