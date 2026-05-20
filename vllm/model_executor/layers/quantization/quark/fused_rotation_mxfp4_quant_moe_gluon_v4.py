#!/usr/bin/env python3
"""
MoE Fused Rotation + MXFP4 Quant + Sorted Scale Scatter — Gluon v4
====================================================================
GEAK iteration 4: 2-kernel design.

K1 (@gluon.jit): MFMA rotation + v_cvt FP4 + scratch write to sorted_scale[m_o+tok, col]
K2 (@triton.jit): flat scatter — tl.arange(0, MAX_Q) matching HIP's vectorized pass

Architecture:
  K1 grid: (ceil(M/BLOCK_M), K//RS)
  K2 grid: (K//RS,)  — same as HIP kernel

K1 writes e8m0 to scratch rows: sorted_scale[m_o + m_base + local_row, col]
K2 reads scratch and writes to MoE shuffle layout: sorted_scale[dst_row, dst_col]

This matches Dense Gluon for K1 (MFMA precision → 100% FP4 match vs bf16 ref),
and matches HIP for K2 scatter (single tl.arange flat vectorized scatter).
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

if not (hasattr(gl, 'amd') and hasattr(gl.amd, 'cdna4')):
    raise ImportError("Gluon cdna4 not available (requires Triton 3.5.x)")

QGROUP = 32
FP4_ELEMS_PER_BYTE = 2
BLOCK_M = 32
NUM_WARPS = 4


# ── K1: Gluon MFMA + FP4 + scratch write ──────────────────────────────────
@gluon.jit
def _gluon_v4_mfma_quant_kernel(
    x_ptr, rot_ptr, fp4_ptr, scratch_ptr,
    M, m_o,
    stride_x_m, stride_rot_r, stride_rot_c,
    stride_fp4_m, stride_sc_m, stride_sc_n,
    RS:       gl.constexpr,
    QG:       gl.constexpr,
    BLOCK_M:  gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """
    K1: MFMA rotation + v_cvt FP4 quantization + scratch write.
    Grid: (ceil(M/BLOCK_M), K//RS)
    Writes e8m0 to scratch_ptr[m_o + m_base + local_row, col_base + c].
    """
    pid_m   = gl.program_id(0)
    pid_rot = gl.program_id(1)

    NUM_QG:  gl.constexpr = RS // QG
    HALF_QG: gl.constexpr = QG // 2

    # ── Layouts ────────────────────────────────────────────────────────────
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

    m_base   = pid_m * BLOCK_M
    col_base = pid_rot * RS
    offs_m   = m_base + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked_mk))
    m_mask   = offs_m < M

    # ── Load x [BLOCK_M, RS] ───────────────────────────────────────────────
    offs_k = gl.arange(0, RS, layout=gl.SliceLayout(0, blocked_mk))
    x_tile = gl.amd.cdna4.buffer_load(
        ptr=x_ptr,
        offsets=offs_m[:, None] * stride_x_m + (col_base + offs_k)[None, :],
        mask=m_mask[:, None],
    )

    # ── Load rotation [RS, RS] → SMEM ──────────────────────────────────────
    smem_rot = gl.allocate_shared_memory(gl.bfloat16, [RS, RS], layout=shared_rot)
    offs_rr = gl.arange(0, RS, layout=gl.SliceLayout(1, blocked_kn))
    offs_rc = gl.arange(0, RS, layout=gl.SliceLayout(0, blocked_kn))
    smem_rot.store(gl.amd.cdna4.buffer_load(
        ptr=rot_ptr,
        offsets=offs_rr[:, None] * stride_rot_r + offs_rc[None, :] * stride_rot_c,
    ))

    # ── MFMA matmul ────────────────────────────────────────────────────────
    acc = gl.zeros((BLOCK_M, RS), gl.float32, layout=mfma_layout)
    acc = gl.amd.cdna4.mfma(gl.convert_layout(x_tile, dot_a), smem_rot.load(layout=dot_b), acc)
    acc = acc.to(gl.bfloat16).to(gl.float32)

    # ── Quantization ───────────────────────────────────────────────────────
    acc_g = gl.reshape(acc, (BLOCK_M, NUM_QG, QG))
    amax  = gl.max(gl.abs(acc_g), axis=-1)

    amax_u32 = amax.to(gl.uint32, bitcast=True)
    amax_u32 = (amax_u32 + 0x400000) & 0xFF800000
    raw_exp  = (amax_u32 >> 23) & 0xFF
    e8m0     = gl.maximum(raw_exp, 2) - 2
    e8m0_u8  = e8m0.to(gl.uint8)

    hw_scale      = (e8m0.to(gl.uint32) << 23).to(gl.float32, bitcast=True)
    hw_scale_full = acc_g * 0.0 + gl.expand_dims(hw_scale, axis=2)

    acc_pairs  = gl.reshape(acc_g, (BLOCK_M, NUM_QG, HALF_QG, 2))
    even_v, odd_v = gl.split(acc_pairs)
    sc_pairs   = gl.reshape(hw_scale_full, (BLOCK_M, NUM_QG, HALF_QG, 2))
    sc_even, _ = gl.split(sc_pairs)

    packed_u32 = tl.inline_asm_elementwise(
        asm="v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3",
        constraints="=v,v,v,v",
        args=[gl.reshape(even_v, (BLOCK_M, NUM_QG * HALF_QG)),
              gl.reshape(odd_v,  (BLOCK_M, NUM_QG * HALF_QG)),
              gl.reshape(sc_even,(BLOCK_M, NUM_QG * HALF_QG))],
        dtype=tl.uint32, is_pure=True, pack=1,
    )
    packed = gl.reshape(packed_u32.to(gl.uint8), (BLOCK_M, RS // 2))

    # ── Store FP4 ──────────────────────────────────────────────────────────
    fp4_store: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4], threads_per_warp=[2, 32],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0],
    )
    packed_s = gl.convert_layout(packed, fp4_store)
    s_m  = m_base + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, fp4_store))
    fp4_c = pid_rot * (RS // 2) + gl.arange(0, RS // 2, layout=gl.SliceLayout(0, fp4_store))
    gl.amd.cdna4.buffer_store(stored_value=packed_s, ptr=fp4_ptr,
        offsets=s_m[:, None] * stride_fp4_m + fp4_c[None, :], mask=(s_m < M)[:, None])

    # ── Write raw e8m0 to scratch ──────────────────────────────────────────
    # scratch_ptr[m_o + m_base + local_row, col_base_q + c]
    sc_store: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1], threads_per_warp=[32, 2],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0],
    )
    sc_s     = gl.convert_layout(e8m0_u8, sc_store)
    sc_m_idx = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, sc_store))
    sc_q_idx = gl.arange(0, NUM_QG,  layout=gl.SliceLayout(0, sc_store))
    sc_col   = pid_rot * NUM_QG + sc_q_idx
    sc_mask  = (m_base + sc_m_idx[:, None]) < M
    tl.store(
        scratch_ptr + (m_o + m_base + sc_m_idx[:, None]) * stride_sc_m
                    + sc_col[None, :] * stride_sc_n,
        sc_s, mask=sc_mask,
    )


# ── K2: Triton scatter with precomputed dst_offset LUT (GEAK iter 7) ───────
# The LUT eliminates int arithmetic in K2 hot loop (was 2.7-4.3µs at M=8-32)
@triton.jit
def _triton_scatter_kernel(
    sorted_ids_ptr, num_valid_ids_ptr, scratch_ptr,
    dst_off_lut_ptr,    # [n_rot, MAX_Q, NUM_QG] int32 byte offset from scratch_ptr
    src_col_off_ptr,    # [n_rot, NUM_QG] int32 src col byte offset
    token_num, m_o,
    stride_sc_m,
    NUM_QG: tl.constexpr,
    MAX_Q:  tl.constexpr,
):
    """K2 with precomputed scatter address LUT.
    Grid: (K // RS,). Pure load/store hot loop, no integer arithmetic.
    """
    pid_rot = tl.program_id(0)
    num_valid = tl.load(num_valid_ids_ptr)

    c = tl.arange(0, NUM_QG)
    q = tl.arange(0, MAX_Q)

    sid = tl.load(sorted_ids_ptr + q, mask=q < num_valid, other=token_num)
    tok = sid & 0xFFFFFF
    q_valid = (q[:, None] < num_valid) & (tok[:, None] < token_num) & (q[:, None] < m_o)

    # Pre-loaded source col byte offsets
    src_col_off = tl.load(src_col_off_ptr + pid_rot * NUM_QG + c)

    # Read from scratch
    val = tl.load(
        scratch_ptr + (m_o + tok)[:, None] * stride_sc_m + src_col_off[None, :],
        mask=q_valid, other=0
    )

    # Pre-loaded full destination offsets
    dst_off = tl.load(dst_off_lut_ptr + pid_rot * MAX_Q * NUM_QG
                      + q[:, None] * NUM_QG + c[None, :])

    tl.store(scratch_ptr + dst_off, val, mask=q_valid)


def _build_scatter_lut(MAX_Q, n_rot, NUM_QG, N_I, TILE_N,
                         stride_sc_m, stride_sc_n, device):
    """Precompute (dst_off, src_col_off) LUT for K2 scatter (CPU side, once)."""
    import numpy as np
    pid_rot = np.arange(n_rot)[:, None, None]
    q = np.arange(MAX_Q)[None, :, None]
    c = np.arange(NUM_QG)[None, None, :]
    col = pid_rot * NUM_QG + c
    pid_n = col >> 3
    n_local = col & 3
    half = (col >> 2) & 1
    q_in = q & 31
    q_half = q_in >> 4
    m_local = q_in & 15
    pid_m_out = q >> 5
    i = q_half + 2 * half
    flat = (((pid_m_out * TILE_N + pid_n) * 4 + n_local) * 16 + m_local) * 4 + i
    dst_row = flat // N_I
    dst_col = flat % N_I
    dst_off = (dst_row * stride_sc_m + dst_col * stride_sc_n).astype(np.int32)
    dst_lut = torch.from_numpy(dst_off).contiguous().to(device)

    pid_rot2 = np.arange(n_rot)[:, None]
    c2 = np.arange(NUM_QG)[None, :]
    src_col_off = ((pid_rot2 * NUM_QG + c2) * stride_sc_n).astype(np.int32)
    src_lut = torch.from_numpy(src_col_off).contiguous().to(device)
    return dst_lut, src_lut


# Cache LUTs per (n_rot, MAX_Q, N_I, TILE_N, stride_sc_m, stride_sc_n)
_lut_cache: dict = {}

def _get_scatter_lut(MAX_Q, n_rot, NUM_QG, N_I, TILE_N, stride_sc_m, stride_sc_n, device):
    key = (MAX_Q, n_rot, NUM_QG, N_I, TILE_N, stride_sc_m, stride_sc_n,
           device.index if device.index is not None else 0)
    cached = _lut_cache.get(key)
    if cached is not None:
        return cached
    luts = _build_scatter_lut(MAX_Q, n_rot, NUM_QG, N_I, TILE_N,
                                stride_sc_m, stride_sc_n, device)
    _lut_cache[key] = luts
    return luts


def _next_pow2(v: int) -> int:
    p = 1
    while p < v:
        p <<= 1
    return p



def fused_rot_quant_moe_sort_gluon_v4(
    x: torch.Tensor,
    rotation: torch.Tensor,
    rotation_size: int,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    m_pad: int,
    fp4_out: torch.Tensor | None = None,
    sorted_scale_out: torch.Tensor | None = None,
    topk: int = 8,
    num_experts: int = 128,
    block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused MoE rotation + MXFP4 quant + sorted scale scatter (Gluon v4).
    2-kernel design: Gluon MFMA+FP4+scratch (K1) + Triton flat scatter (K2).

    GEAK iter 6: Smart MAX_Q = next_pow2(min(M*topk, E)*block_size)
                 — much smaller than m_pad for small M, gives 3-4µs speedup.
    """
    assert x.ndim == 2
    M, K = x.shape
    assert K % rotation_size == 0
    RS   = rotation_size
    QG   = QGROUP
    n_i  = K // QG
    num_qg = RS // QG   # scale columns per rotation block
    m_o  = sorted_ids.shape[0]

    if fp4_out is None:
        fp4_out = torch.empty((M, K // FP4_ELEMS_PER_BYTE),
                              dtype=torch.uint8, device=x.device)

    # Buffer: m_pad rows (scatter output) + M rows scratch (at offset m_o)
    scratch_rows = ((M + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    total_rows   = max(m_pad, m_o) + scratch_rows
    if sorted_scale_out is None:
        sorted_scale_out = torch.zeros((total_rows, n_i),
                                       dtype=torch.uint8, device=x.device)
    else:
        # Ensure buffer is large enough
        assert sorted_scale_out.shape[0] >= total_rows

    tile_n   = triton.cdiv(n_i, 8)
    n_rot    = K // RS

    # GEAK iter 6: Smart MAX_Q upper bound.
    nv_upper = min(M * topk, num_experts) * block_size
    max_q    = _next_pow2(nv_upper) if nv_upper > 0 else 64
    # Adaptive num_warps for K2 (M=1,8 → nw=4; else nw=8)
    k2_warps = 4 if M in (1, 8) else 8

    # GEAK iter 7: Precompute scatter LUT (cached per shape)
    dst_lut, src_lut = _get_scatter_lut(max_q, n_rot, num_qg, n_i, tile_n,
                                          sorted_scale_out.stride(0),
                                          sorted_scale_out.stride(1),
                                          x.device)

    # K1: MFMA + FP4 + scratch write
    grid_k1 = (triton.cdiv(M, BLOCK_M), n_rot)
    _gluon_v4_mfma_quant_kernel[grid_k1](
        x, rotation, fp4_out, sorted_scale_out,
        M, m_o,
        x.stride(0), rotation.stride(0), rotation.stride(1),
        fp4_out.stride(0),
        sorted_scale_out.stride(0), sorted_scale_out.stride(1),
        RS=RS, QG=QG, BLOCK_M=BLOCK_M,
        NUM_WARPS=NUM_WARPS, num_warps=NUM_WARPS,
    )

    # K2: Pure load/store with precomputed LUT (no integer arithmetic in hot loop)
    grid_k2 = (n_rot,)
    _triton_scatter_kernel[grid_k2](
        sorted_ids, num_valid_ids, sorted_scale_out,
        dst_lut, src_lut,
        token_num, m_o,
        sorted_scale_out.stride(0),
        NUM_QG=num_qg, MAX_Q=max_q,
        num_warps=k2_warps,
    )

    return fp4_out, sorted_scale_out[:m_pad]
