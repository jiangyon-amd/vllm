"""
Gluon rotation kernels for MoE hidden states.

Single kernel parameterized by k_width (constexpr):
  k_width=4  → used for M<4  (V2 behaviour, less MFMA pressure at tiny M)
  k_width=8  → used for M>=4 (V3 behaviour, better throughput via wider dot)

Dispatch (called from aiter_rotation_patch.py):
  gluon_moe_rotation(x, rotation, rotation_size, M)
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

BLOCK_M = 32
NUM_WARPS = 4
NUM_STAGES = 3


@gluon.jit
def _gluon_rotation_kernel(
    x_ptr, rot_ptr, y_ptr,
    M, sx_m, sr_r, sr_c, sy_m,
    RS:        gl.constexpr,
    BLOCK_M:   gl.constexpr,
    NUM_WARPS: gl.constexpr,
    K_WIDTH:   gl.constexpr,  # 4 for M<4, 8 for M>=4
):
    """Parallel rotation: one CTA per (M-tile, K-block). Grid: (ceil(M/BM), K//RS)."""
    pid_m = gl.program_id(0)
    pid_rot = gl.program_id(1)

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16], transposed=True,
        warps_per_cta=[2, 2], tiles_per_warp=[1, 4])
    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, K_WIDTH], threads_per_warp=[16, 4],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0])
    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[K_WIDTH, 1], threads_per_warp=[4, 16],
        warps_per_cta=[1, NUM_WARPS], order=[0, 1])
    shared_rot: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[1, 0])
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=K_WIDTH)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=K_WIDTH)
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


def gluon_moe_rotation(
    x: torch.Tensor,
    rotation: torch.Tensor,
    rotation_size: int,
) -> torch.Tensor:
    """Apply blockwise rotation to MoE hidden states using Gluon MFMA kernel.

    Args:
        x: Input [M, K] bfloat16.
        rotation: Rotation matrix [RS, RS] bfloat16.
        rotation_size: Block size RS. K must be divisible by RS.

    Returns:
        Rotated tensor [M, K] bfloat16 (pre-allocated buffer, CUDAGraph-safe).
    """
    assert x.ndim == 2 and rotation.ndim == 2
    M, K = x.shape
    RS = rotation_size
    assert K % RS == 0

    out = _get_y_buf((M, K), torch.bfloat16, x.device)
    grid = (triton.cdiv(M, BLOCK_M), K // RS)

    # k_width=8 wins for M>=4 (better MFMA throughput); k_width=4 for M<4
    k_width = 8 if M >= 4 else 4
    _gluon_rotation_kernel[grid](
        x, rotation, out,
        M, x.stride(0), rotation.stride(0), rotation.stride(1), out.stride(0),
        RS=RS, BLOCK_M=BLOCK_M, NUM_WARPS=NUM_WARPS, K_WIDTH=k_width,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out
