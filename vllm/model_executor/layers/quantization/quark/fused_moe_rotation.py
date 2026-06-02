"""
MoE rotation feature flags and Gluon kernel activation.

Environment variables:
  VLLM_FUSED_ROTATION=1            → enable both Dense Gluon + MoE Gluon (recommended)
  VLLM_FUSED_ROTATION=0            → disable both
  VLLM_MOE_FORCE_GLUON_ROTATION=1  → enable MoE Gluon only (legacy)
"""

import logging
import os
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MoE Gluon monkey-patch: replaces torch.matmul rotation in aiter's
# fused_moe_2stages with a Gluon MFMA kernel (+0.8-2.2µs for M=4-32).
# ---------------------------------------------------------------------------
_PATCH_APPLIED = False


def _apply_gluon_rotation_patch() -> None:
    """Monkey-patch aiter.fused_moe_2stages to use Gluon rotation kernel."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    import aiter.fused_moe as afm
    from vllm.model_executor.layers.quantization.quark.gluon_rotation_moe import (
        gluon_moe_rotation,
    )

    _orig = afm.fused_moe_2stages

    def _patched(*args, **kwargs):
        rotation = kwargs.get("rotation")
        rotation_size = kwargs.get("rotation_size", 0)
        if rotation is None or rotation_size <= 0:
            return _orig(*args, **kwargs)
        hidden = args[0]
        if hidden.dim() == 2 and hidden.dtype == torch.bfloat16:
            rotated = gluon_moe_rotation(hidden, rotation, rotation_size)
            return _orig(*((rotated,) + args[1:]),
                         **{**kwargs, "rotation": None, "rotation_size": 0})
        return _orig(*args, **kwargs)

    afm.fused_moe_2stages = _patched
    _PATCH_APPLIED = True
    logger.info("Applied Gluon rotation patch to aiter.fused_moe_2stages")


def _is_moe_rotation_enabled() -> bool:
    """True when MoE Gluon rotation is requested via env vars."""
    unified = os.environ.get("VLLM_FUSED_ROTATION", "").strip().lower()
    if unified in ("1", "true"):
        return True
    if unified in ("0", "false"):
        return False
    return os.getenv("VLLM_MOE_FORCE_GLUON_ROTATION", "0") == "1"


# Apply patch eagerly at import time if enabled.
if _is_moe_rotation_enabled():
    try:
        _apply_gluon_rotation_patch()
    except Exception as e:
        logger.warning("Failed to apply MoE Gluon rotation patch: %s", e)

# True when aiter supports passing rotation through fused_moe_2stages.
try:
    from vllm._aiter_ops import rocm_aiter_ops as _aiter_ops
    _has_moe_rot_quant: bool = (
        _is_moe_rotation_enabled() and hasattr(_aiter_ops, 'fused_moe')
    )
except Exception:
    _has_moe_rot_quant = False

# Keep FUSED_MOE_ROTATION as alias for backward compat with qwen3_moe.py import
FUSED_MOE_ROTATION = _has_moe_rot_quant
