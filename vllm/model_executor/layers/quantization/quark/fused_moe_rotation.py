"""
MoE rotation feature flags and Gluon kernel activation.

Environment variables:
  VLLM_FUSED_ROTATION=1            → enable both Dense Gluon + MoE Gluon (recommended)
  VLLM_FUSED_ROTATION=0            → disable both
  VLLM_MOE_FUSED_ROTATION=0        → disable MoE fused rotation entirely
  VLLM_MOE_FORCE_GLUON_ROTATION=1  → enable MoE Gluon only (legacy)
"""

import logging
import os
import torch

logger = logging.getLogger(__name__)

# Whether MoE fused rotation+quant pipeline is active (aiter 3-in-1 path).
FUSED_MOE_ROTATION: bool = os.getenv("VLLM_MOE_FUSED_ROTATION", "1") == "1"

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


def _is_moe_gluon_rotation_enabled() -> bool:
    unified = os.environ.get("VLLM_FUSED_ROTATION", "").strip().lower()
    if unified in ("1", "true"):
        return True
    if unified in ("0", "false"):
        return False
    return os.getenv("VLLM_MOE_FORCE_GLUON_ROTATION", "0") == "1"


if _is_moe_gluon_rotation_enabled():
    try:
        _apply_gluon_rotation_patch()
    except Exception as e:
        logger.warning("Failed to apply MoE Gluon rotation patch: %s", e)

try:
    from vllm.model_executor.layers.quantization.quark.fused_rotation_mxfp4_quant_moe_sort import (  # noqa: F401
        fused_rotation_mxfp4_quant_moe_sort,
    )
    _has_moe_rot_quant = True
except Exception:
    _has_moe_rot_quant = False
