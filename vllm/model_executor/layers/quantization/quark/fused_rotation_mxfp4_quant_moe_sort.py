"""
Fused Rotation + MXFP4 Quant + MoE Sort — dispatcher.

Dispatch policy (MI355X CUDAGraph benchmark, Qwen3-30B-A3B, topk=4):
  M == 1:  Triton-legacy kernel (3.3µs) — faster than aiter Gluon-M1 (4.8µs)
  M > 1:   aiter unified API (HIP MFMA 3-in-1 via AITER_MOE_HIP_MFMA=1)
"""

import logging
import os
import triton
import torch
from aiter.utility import dtypes

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import the unified aiter API — single entry point for M>1 dispatch
# ---------------------------------------------------------------------------
try:
    from aiter.ops.triton.fused_rot_quant_moe_sort import (
        fused_rot_quant_moe_sort as _aiter_fused_fn,
        warmup_kernels as _aiter_warmup_fn,
    )
    _HAS_AITER_UNIFIED = True
except ImportError:
    _HAS_AITER_UNIFIED = False
    _aiter_fused_fn = None
    _aiter_warmup_fn = None
    logger.warning("aiter fused_rot_quant_moe_sort not available")

# ---------------------------------------------------------------------------
# Triton-legacy M=1 kernel — fastest for single-token decode (3.3µs)
# aiter-auto uses Gluon-M1 (4.8µs) at M=1; this bypasses that.
# ---------------------------------------------------------------------------
try:
    from aiter.ops.triton._triton_kernels.fused_rot_quant_moe_sort import (
        _rot_quant_sort_m1_kernel as _triton_m1_kernel,
    )
    _HAS_TRITON_M1 = True
except ImportError:
    _triton_m1_kernel = None
    _HAS_TRITON_M1 = False
    logger.warning("aiter Triton M=1 MoE kernel not available, will use aiter-auto")

# Allow disabling the M=1 Triton-legacy fast path via env var.
# Some kernels (e.g. aiter-unified) may now be faster on newer GPUs.
if os.getenv("VLLM_MOE_DISABLE_TRITON_M1", "0") == "1":
    _HAS_TRITON_M1 = False
    logger.info("VLLM_MOE_DISABLE_TRITON_M1=1 → forcing aiter-unified for all M")

# Track which K values have been warmed up (Triton JIT pre-compilation).
_warmed_up_ks: set[int] = set()

# ---------------------------------------------------------------------------
# Buffer cache for M=1 Triton path (CUDAGraph compatibility).
# Pre-allocate fp4/scale output tensors per (K, m_pad) so CUDAGraph
# captures fixed-address writes — no dynamic allocation inside graph.
# ---------------------------------------------------------------------------
_triton_m1_buf_cache: dict = {}

def _get_triton_m1_bufs(K: int, m_pad: int, n_i: int, device: torch.device):
    key = (K, m_pad, device.index if device.index is not None else 0)
    cached = _triton_m1_buf_cache.get(key)
    if cached is not None:
        return cached
    fp4_buf = torch.empty((1, K // 2), dtype=torch.uint8, device=device)
    sc_buf  = torch.empty((m_pad, n_i), dtype=torch.uint8, device=device)
    _triton_m1_buf_cache[key] = (fp4_buf, sc_buf)
    return fp4_buf, sc_buf


# ---------------------------------------------------------------------------
# Register as custom op for CUDAGraph compatibility
# ---------------------------------------------------------------------------
_has_custom_op = False
try:
    from vllm.utils.torch_utils import direct_register_custom_op
    from vllm.platforms import current_platform

    def _fused_rot_quant_moe_sort_op(
        x: torch.Tensor,
        rotation: torch.Tensor,
        rotation_size: int,
        sorted_ids: torch.Tensor,
        num_valid_ids: torch.Tensor,
        token_num: int,
        topk: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _fused_rot_quant_moe_sort_impl(
            x, rotation, rotation_size,
            sorted_ids, num_valid_ids,
            token_num, topk, block_size,
        )

    def _fused_rot_quant_moe_sort_fake(
        x: torch.Tensor,
        rotation: torch.Tensor,
        rotation_size: int,
        sorted_ids: torch.Tensor,
        num_valid_ids: torch.Tensor,
        token_num: int,
        topk: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        M, K = x.shape
        QG = 32
        n_i = K // QG
        m_o = sorted_ids.shape[0]
        m_pad = ((m_o + 31) // 32) * 32
        return (
            torch.empty((M, K // 2), dtype=dtypes.fp4x2, device=x.device),
            torch.empty((m_pad, n_i), dtype=dtypes.fp8_e8m0, device=x.device),
        )

    direct_register_custom_op(
        op_name="fused_rotation_mxfp4_quant_moe_sort",
        op_func=_fused_rot_quant_moe_sort_op,
        mutates_args=[],
        fake_impl=_fused_rot_quant_moe_sort_fake,
        dispatch_key=current_platform.dispatch_key,
    )
    _has_custom_op = True
    logger.info("Registered fused_rotation_mxfp4_quant_moe_sort custom op")
except Exception as e:
    logger.warning("Failed to register custom op: %s", e)


# ---------------------------------------------------------------------------
# Core implementation — delegates to aiter unified API
# ---------------------------------------------------------------------------
def _run_triton_m1(
    x: torch.Tensor,
    rotation: torch.Tensor,
    RS: int,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    block_size: int,
    m_o: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-legacy M=1 decode kernel — fastest for single-token decode.
    Benchmark: 3.3µs vs Gluon-M1 4.8µs vs HIP 5.2µs (MI355X CUDAGraph).
    Uses num_warps=1 for minimal wave occupancy at M=1.
    CUDAGraph-safe: pre-allocated buffers via _get_triton_m1_bufs().
    m_o must be a Python int (computed before graph capture, not from tensor.item()).
    """
    _, K = x.shape
    QG = 32
    n_i = K // QG
    m_pad = ((sorted_ids.shape[0] + block_size - 1) // block_size) * block_size

    tile_n = triton.cdiv(n_i, 8)
    max_q = 64 if m_pad <= 64 else (128 if m_pad <= 128 else 256)
    grid = (K // RS,)

    # CUDAGraph-safe: cached pre-allocated output buffers
    fp4_out, sc_out = _get_triton_m1_bufs(K, m_pad, n_i, x.device)

    _triton_m1_kernel[grid](
        x, rotation, fp4_out,
        sorted_ids, num_valid_ids, sc_out,
        x.stride(1),
        rotation.stride(0), rotation.stride(1),
        fp4_out.stride(0),
        sc_out.stride(0), sc_out.stride(1),
        token_num=token_num, m_o=m_o,
        N_I=n_i, TILE_N=tile_n, RS=RS, QG=QG, MAX_Q=max_q,
        num_warps=1,
    )
    return fp4_out.view(dtypes.fp4x2), sc_out.view(dtypes.fp8_e8m0)


def _fused_rot_quant_moe_sort_impl(
    x: torch.Tensor,
    rotation: torch.Tensor,
    rotation_size: int,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    topk: int,
    block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch: M==1 → Triton-legacy (3.3µs); M>1 → aiter unified (HIP MFMA)."""
    RS = rotation_size
    M, K = x.shape

    # M=1 decode: use faster Triton-legacy kernel (bypasses aiter Gluon-M1)
    # 3.3µs vs Gluon-M1 4.8µs — 31% faster for single-token decode.
    # m_o is computed once in Python before CUDAGraph capture (token_num*topk).
    if M == 1 and _HAS_TRITON_M1:
        m_o = token_num * topk   # pure Python int — safe inside CUDAGraph
        n_i = K // 32
        m_pad = ((sorted_ids.shape[0] + block_size - 1) // block_size) * block_size
        # Pre-warm buffers (called during vLLM warmup, before graph capture)
        _get_triton_m1_bufs(K, m_pad, n_i, x.device)
        return _run_triton_m1(x, rotation, RS, sorted_ids, num_valid_ids,
                              token_num, block_size, m_o)

    # M>1: aiter unified (HIP MFMA 3-in-1 for M=2~64, Gluon-kw8 for M>=128)
    if not _HAS_AITER_UNIFIED:
        raise RuntimeError(
            "aiter fused_rot_quant_moe_sort not available. "
            "Install aiter with HIP MFMA support."
        )

    if K not in _warmed_up_ks and _aiter_warmup_fn is not None:
        _aiter_warmup_fn(K=K, topk=topk, device=x.device)
        _warmed_up_ks.add(K)

    return _aiter_fused_fn(
        x, rotation, RS,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token_num,
        block_size=block_size,
    )


# ---------------------------------------------------------------------------
# Public API (backward compatible)
# ---------------------------------------------------------------------------
def fused_rotation_mxfp4_quant_moe_sort(
    x: torch.Tensor,
    rotation: torch.Tensor,
    rotation_size: int,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    topk: int,
    block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused rotation + MXFP4 quant + MoE sort. CUDAGraph compatible."""
    if _has_custom_op:
        return torch.ops.vllm.fused_rotation_mxfp4_quant_moe_sort(
            x, rotation, rotation_size,
            sorted_ids, num_valid_ids,
            token_num, topk, block_size,
        )
    return _fused_rot_quant_moe_sort_impl(
        x, rotation, rotation_size,
        sorted_ids, num_valid_ids,
        token_num, topk, block_size,
    )
