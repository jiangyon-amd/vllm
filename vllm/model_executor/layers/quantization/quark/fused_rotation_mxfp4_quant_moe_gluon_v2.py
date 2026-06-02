#!/usr/bin/env python3
"""
MoE Fused Rotation + MXFP4 Quant + Sorted Scale Scatter — Gluon v2

A NEW MoE 3-in-1 Gluon kernel that mirrors the structure of the proven
dense Gluon kernel (`fused_rotation_quant_gluon.py`):
  - Same MFMA layout (CDNA4 v4 16x16, transposed)
  - Same BLOCK_M=32, NUM_WARPS=4, k_width=4
  - Same buffer_load + shared-memory rotation
  - Same v_cvt_scalef32_pk_fp4_f32 for MXFP4
  - **NEW**: replace the dense scale-store with MoE sorted-scatter

Goal: deliver the dense kernel's E2E TPOT improvement (~−2%) on MoE models.
"""

import torch
import triton
from triton import language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


# Same constants as dense Gluon
QGROUP = 32
FP4_ELEMS_PER_BYTE = 2
BLOCK_M = 32
NUM_WARPS = 4


@gluon.jit
def _gluon_v2_rot_quant_moe_sort_kernel(
    x_ptr, rot_ptr, fp4_ptr, scale_ptr,
    sorted_ids_ptr, num_valid_ids_ptr,
    M, K, m_o, max_q,
    stride_x_m, stride_rot_r, stride_rot_c,
    stride_fp4_m, stride_sc_m,
    RS: gl.constexpr,
    QG: gl.constexpr,
    BLOCK_M: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    N_I: gl.constexpr,    # K // QG  (constexpr for fast div/mod)
    TILE_N: gl.constexpr, # cdiv(N_I, 8)
    MAX_Q: gl.constexpr,  # static-unrolled scatter loop bound
):
    """
    Tile work:
      - Each program handles a (BLOCK_M, RS) tile of input x.
      - Rotates the tile via MFMA against R [RS, RS].
      - MXFP4-quantizes via inline asm (8 fp4 packed).
      - Writes packed fp4 to fp4_ptr (row-major by token).
      - For each output position q in sorted_ids, scatters the scales
        into the AMD MoE-shuffled scale buffer.

    Designed to work for BOTH M=1 (decode) and M>1 (prefill).
    """
    pid_m   = gl.program_id(0)
    pid_rot = gl.program_id(1)

    # ======== Layouts (mirror dense) ========
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

    # ======== Load x [BLOCK_M, RS] via buffer_load (mirror dense) ========
    offs_k = gl.arange(0, RS, layout=gl.SliceLayout(0, blocked_mk))
    x_offs = offs_m[:, None] * stride_x_m + (col_base + offs_k)[None, :]
    x_tile = gl.amd.cdna4.buffer_load(ptr=x_ptr, offsets=x_offs, mask=m_mask[:, None])

    # ======== Load rotation [RS, RS] ========
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
    acc = acc.to(gl.bfloat16).to(gl.float32)

    # ======== Quantization (same as dense) ========
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

    # ======== Store FP4 (row-major, mirror dense) ========
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

    # ======== MoE sorted scatter for scales ========
    # Dump per-token scales to a "raw" temp row first (same trick as M=1 kernel)
    sc_store2: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1], threads_per_warp=[32, 2],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0],
    )
    sc_s = gl.convert_layout(e8m0_u8, sc_store2)
    sc_tmp_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, sc_store2))
    sc_tmp_c = pid_rot * NUM_QG + gl.arange(0, NUM_QG, layout=gl.SliceLayout(0, sc_store2))
    sc_tmp_mask = (m_base + sc_tmp_m) < M

    # Write raw e8m0 to scratch rows starting at m_o
    tl.store(scale_ptr + (m_o + m_base + sc_tmp_m[:, None]) * stride_sc_m
             + sc_tmp_c[None, :],
             sc_s, mask=sc_tmp_mask[:, None])

    # Scatter loop: for each output position q in sorted_ids,
    # find the source token, look up its scale, write to AMD-shuffled position.
    num_valid = tl.load(num_valid_ids_ptr)

    sc_scatter: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1], threads_per_warp=[16, 4],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0],
    )

    for qb in tl.static_range(0, MAX_Q // 64):
        q = qb * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, sc_scatter))
        c = gl.arange(0, NUM_QG, layout=gl.SliceLayout(0, sc_scatter))
        col = pid_rot * NUM_QG + c

        sid = tl.load(sorted_ids_ptr + q[:, None] + c[None, :] * 0,
                       mask=q[:, None] < num_valid, other=M)
        tok = sid & 0xFFFFFF

        # Only scatter if (a) within sorted_ids range, (b) token belongs to our M tile,
        #                 (c) within unrolled bound.
        in_tile = (tok >= m_base) & (tok < (m_base + BLOCK_M))
        q_valid = (q[:, None] < num_valid) & in_tile & (q[:, None] < m_o)

        # AMD MoE shuffle layout
        pid_n = col >> 3
        n_local = col & 3
        half = (col >> 2) & 1
        q_in = q & 31
        q_half = q_in >> 4
        m_local = q_in & 15
        pid_m_out = q >> 5
        i = q_half[:, None] + half[None, :] * 2
        flat = (((pid_m_out[:, None] * TILE_N + pid_n[None, :]) * 4
                 + n_local[None, :]) * 16 + m_local[:, None]) * 4 + i
        dst_row = flat // N_I
        dst_col = flat % N_I

        # Read the source token's scale from scratch row
        src_row = m_o + tok
        val = tl.load(scale_ptr + src_row * stride_sc_m + col[None, :],
                      mask=q_valid)
        tl.store(scale_ptr + dst_row * stride_sc_m + dst_col, val, mask=q_valid)


def fused_rot_quant_moe_sort_gluon_v2(
    x: torch.Tensor,
    rotation: torch.Tensor,
    rotation_size: int,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    block_size: int = 32,
    fp4_out: torch.Tensor | None = None,
    sorted_scale_out: torch.Tensor | None = None,
):
    """
    3-in-1 Gluon MoE: rotation + MXFP4 quant + AMD-shuffled sorted scatter.
    Compatible API with aiter's `fused_rot_quant_moe_sort`.
    """
    assert x.ndim == 2
    M, K = x.shape
    assert K % rotation_size == 0
    RS = rotation_size
    QG = QGROUP
    n_i = K // QG

    # m_o = number of valid output positions = sum of expert assignments
    # (= token_num * topk; we get it from sorted_ids dimensions if needed)
    m_o = M * (sorted_ids.shape[0] // max(M, 1)) if M > 0 else 0
    # MAX_Q for the static-unrolled scatter loop — must be multiple of 64.
    m_pad = sorted_ids.shape[0]
    MAX_Q = ((m_pad + 63) // 64) * 64

    if fp4_out is None:
        fp4_out = torch.empty((M, K // FP4_ELEMS_PER_BYTE),
                              dtype=torch.uint8, device=x.device)
    if sorted_scale_out is None:
        # Need scratch rows starting at row m_o for raw storage.
        sm_padded = ((m_pad + 255) // 256) * 256
        scratch = max(sm_padded, m_o + ((M + 31) // 32) * 32)
        sorted_scale_out = torch.empty((scratch, n_i),
                                       dtype=torch.uint8, device=x.device)

    grid = (triton.cdiv(M, BLOCK_M), K // RS)
    tile_n = triton.cdiv(n_i, 8)

    _gluon_v2_rot_quant_moe_sort_kernel[grid](
        x, rotation, fp4_out, sorted_scale_out,
        sorted_ids, num_valid_ids,
        M, K, m_o, MAX_Q,
        x.stride(0), rotation.stride(0), rotation.stride(1),
        fp4_out.stride(0), sorted_scale_out.stride(0),
        RS=RS, QG=QG, BLOCK_M=BLOCK_M,
        NUM_WARPS=NUM_WARPS, num_warps=NUM_WARPS,
        N_I=n_i, TILE_N=tile_n, MAX_Q=MAX_Q,
    )
    # Return with aiter-expected dtypes (fp4x2 packed + fp8_e8m0 scales)
    return (fp4_out.view(torch.float4_e2m1fn_x2),
            sorted_scale_out.view(torch.float8_e8m0fnu))
