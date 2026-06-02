#!/usr/bin/env python3
"""
Gluon Rotation Kernel V2 — exploits "K is small, no inter-block accumulation".

Key insight from user:
- rotation_size (K=128) is small
- Each K-block is INDEPENDENT (no accumulation across blocks)
- Don't need full matmul infrastructure for K-accumulation

Optimizations:
1. PERSISTENT kernel: ONE CTA processes ALL K-blocks for its M-slice
   - Load rotation ONCE into SMEM, reuse for all blocks
   - Eliminates redundant rotation loads (was 16x per CTA × 16 CTAs)
2. NO bf16 round-trip on output (rotation precision doesn't need it)
3. Output bf16 directly from FP32 accumulator
"""
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

BLOCK_M = 32
NUM_WARPS = 4


@gluon.jit
def _gluon_rotation_v2_persistent_kernel(
    x_ptr, rot_ptr, y_ptr,
    M, sx_m, sr_r, sr_c, sy_m,
    RS:           gl.constexpr,
    BLOCK_M:      gl.constexpr,
    NUM_WARPS:    gl.constexpr,
    N_K_BLOCKS:   gl.constexpr,  # = K // RS
):
    """Persistent rotation: ONE CTA does ALL K-blocks for its M-slice.
    Grid: (ceil(M/BLOCK_M),)  -- no pid_rot dimension!
    """
    pid_m = gl.program_id(0)

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16], transposed=True,
        warps_per_cta=[2, 2], tiles_per_warp=[1, 4])
    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4], threads_per_warp=[16, 4],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0])
    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[4, 1], threads_per_warp=[4, 16],
        warps_per_cta=[1, NUM_WARPS], order=[0, 1])
    shared_rot: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[1, 0])
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=4)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=4)
    store_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4], threads_per_warp=[2, 32],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0])

    m_base = pid_m * BLOCK_M
    offs_m = m_base + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked_mk))
    m_mask = offs_m < M

    # === Load rotation ONCE, reuse for all K-blocks ===
    smem_rot = gl.allocate_shared_memory(gl.bfloat16, [RS, RS], layout=shared_rot)
    offs_rr = gl.arange(0, RS, layout=gl.SliceLayout(1, blocked_kn))
    offs_rc = gl.arange(0, RS, layout=gl.SliceLayout(0, blocked_kn))
    smem_rot.store(gl.amd.cdna4.buffer_load(ptr=rot_ptr,
        offsets=offs_rr[:, None] * sr_r + offs_rc[None, :] * sr_c))

    # Load rotation into dot_b layout ONCE - reused across K-blocks
    rot_dot_b = smem_rot.load(layout=dot_b)

    # === Process all K-blocks sequentially (rotation reused) ===
    for k_block in tl.static_range(0, N_K_BLOCKS):
        col_base = k_block * RS

        # Load x[BLOCK_M, RS] for this K-block
        offs_k = gl.arange(0, RS, layout=gl.SliceLayout(0, blocked_mk))
        x_tile = gl.amd.cdna4.buffer_load(ptr=x_ptr,
            offsets=offs_m[:, None] * sx_m + (col_base + offs_k)[None, :],
            mask=m_mask[:, None])

        # MFMA using already-loaded rotation
        acc = gl.zeros((BLOCK_M, RS), gl.float32, layout=mfma_layout)
        acc = gl.amd.cdna4.mfma(gl.convert_layout(x_tile, dot_a),
                                 rot_dot_b, acc)

        # Cast to bf16 (direct from f32, no round-trip)
        y = acc.to(gl.bfloat16)

        # Store
        y_s = gl.convert_layout(y, store_layout)
        s_m_st = m_base + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, store_layout))
        y_c = col_base + gl.arange(0, RS, layout=gl.SliceLayout(0, store_layout))
        gl.amd.cdna4.buffer_store(stored_value=y_s, ptr=y_ptr,
            offsets=s_m_st[:, None] * sy_m + y_c[None, :], mask=(s_m_st < M)[:, None])


@gluon.jit
def _gluon_rotation_v2_parallel_kernel(
    x_ptr, rot_ptr, y_ptr,
    M, sx_m, sr_r, sr_c, sy_m,
    RS:        gl.constexpr,
    BLOCK_M:   gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Parallel rotation: SAME as original but with output cast optimization.
    Grid: (ceil(M/BLOCK_M), K//RS)  -- same as original
    """
    pid_m = gl.program_id(0)
    pid_rot = gl.program_id(1)

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16], transposed=True,
        warps_per_cta=[2, 2], tiles_per_warp=[1, 4])
    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4], threads_per_warp=[16, 4],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0])
    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[4, 1], threads_per_warp=[4, 16],
        warps_per_cta=[1, NUM_WARPS], order=[0, 1])
    shared_rot: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[1, 0])
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=4)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=4)
    store_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4], threads_per_warp=[2, 32],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0])

    m_base = pid_m * BLOCK_M
    col_base = pid_rot * RS
    offs_m = m_base + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked_mk))
    m_mask = offs_m < M

    offs_k = gl.arange(0, RS, layout=gl.SliceLayout(0, blocked_mk))
    x_tile = gl.amd.cdna4.buffer_load(ptr=x_ptr,
        offsets=offs_m[:, None] * sx_m + (col_base + offs_k)[None, :],
        mask=m_mask[:, None])

    smem_rot = gl.allocate_shared_memory(gl.bfloat16, [RS, RS], layout=shared_rot)
    offs_rr = gl.arange(0, RS, layout=gl.SliceLayout(1, blocked_kn))
    offs_rc = gl.arange(0, RS, layout=gl.SliceLayout(0, blocked_kn))
    smem_rot.store(gl.amd.cdna4.buffer_load(ptr=rot_ptr,
        offsets=offs_rr[:, None] * sr_r + offs_rc[None, :] * sr_c))

    acc = gl.zeros((BLOCK_M, RS), gl.float32, layout=mfma_layout)
    acc = gl.amd.cdna4.mfma(gl.convert_layout(x_tile, dot_a),
                             smem_rot.load(layout=dot_b), acc)
    # Direct f32 → bf16, NO round-trip via f32
    y = acc.to(gl.bfloat16)

    y_s = gl.convert_layout(y, store_layout)
    s_m = m_base + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, store_layout))
    y_c = col_base + gl.arange(0, RS, layout=gl.SliceLayout(0, store_layout))
    gl.amd.cdna4.buffer_store(stored_value=y_s, ptr=y_ptr,
        offsets=s_m[:, None] * sy_m + y_c[None, :], mask=(s_m < M)[:, None])


_y_cache: dict = {}

def _get_y_buf(shape, dtype, device):
    key = (shape, dtype, device.index if device.index is not None else 0)
    if key not in _y_cache:
        _y_cache[key] = torch.empty(shape, dtype=dtype, device=device)
    return _y_cache[key]


def gluon_rotation_v2(
    x: torch.Tensor,
    rotation: torch.Tensor,
    rotation_size: int = None,
    out: torch.Tensor = None,
    persistent: bool = True,
) -> torch.Tensor:
    """V2 rotation kernel. persistent=True uses 1 CTA per M-slice (loads rotation once)."""
    assert x.ndim == 2 and rotation.ndim == 2
    M, K = x.shape
    RS = rotation_size if rotation_size is not None else rotation.shape[0]
    assert K % RS == 0
    if out is None:
        out = _get_y_buf((M, K), torch.bfloat16, x.device)

    if persistent:
        grid = (triton.cdiv(M, BLOCK_M),)
        _gluon_rotation_v2_persistent_kernel[grid](
            x, rotation, out,
            M, x.stride(0), rotation.stride(0), rotation.stride(1), out.stride(0),
            RS=RS, BLOCK_M=BLOCK_M, NUM_WARPS=NUM_WARPS, N_K_BLOCKS=K // RS,
            num_warps=NUM_WARPS,
        )
    else:
        grid = (triton.cdiv(M, BLOCK_M), K // RS)
        _gluon_rotation_v2_parallel_kernel[grid](
            x, rotation, out,
            M, x.stride(0), rotation.stride(0), rotation.stride(1), out.stride(0),
            RS=RS, BLOCK_M=BLOCK_M, NUM_WARPS=NUM_WARPS,
            num_warps=NUM_WARPS,
        )
    return out
