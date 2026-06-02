"""
MoE fused-rotation feature flags and kernel activation.

Unified switch (recommended):
  VLLM_FUSED_ROTATION=1  → enable both Dense Gluon + MoE Gluon kernels
  VLLM_FUSED_ROTATION=0  → disable both

Legacy switches (still supported):
  VLLM_MOE_FUSED_ROTATION=0        → disable MoE rotation pipeline entirely
  VLLM_MOE_FORCE_GLUON_ROTATION=1  → enable MoE Gluon only (independent of Dense)

MoE Gluon activation priority:
  1. VLLM_FUSED_ROTATION=1  (unified, recommended)
  2. VLLM_MOE_FORCE_GLUON_ROTATION=1  (legacy, MoE-only)
  Both activate the Gluon rotation monkey-patch on aiter.fused_moe_2stages.
"""

import os

FUSED_MOE_ROTATION = os.getenv("VLLM_MOE_FUSED_ROTATION", "1") == "1"

# Default to HIP MFMA kernel (4x faster than Gluon Triton kernels for prefill).
# User can override with AITER_MOE_HIP_MFMA=0 to fall back to Triton.
os.environ.setdefault("AITER_MOE_HIP_MFMA", "1")


def _is_moe_gluon_rotation_enabled() -> bool:
    """Check whether MoE Gluon rotation patch should be applied.

    Priority:
      VLLM_FUSED_ROTATION=1            → True  (unified switch)
      VLLM_FUSED_ROTATION=0            → False (unified disable)
      VLLM_MOE_FORCE_GLUON_ROTATION=1  → True  (legacy MoE-only switch)
    """
    unified = os.environ.get("VLLM_FUSED_ROTATION", "").strip().lower()
    if unified in ("1", "true"):
        return True
    if unified in ("0", "false"):
        return False
    # Legacy switch
    return os.getenv("VLLM_MOE_FORCE_GLUON_ROTATION", "0") == "1"


# Apply MoE Gluon monkey-patch if enabled
if _is_moe_gluon_rotation_enabled():
    try:
        from vllm.model_executor.layers.quantization.quark.aiter_rotation_patch import (
            apply_gluon_rotation_patch,
        )
        apply_gluon_rotation_patch()
    except Exception as e:
        import logging
        logging.warning(f"Failed to apply MoE Gluon rotation patch: {e}")

try:
    # Validate that the fused MoE rotation implementation is importable.
    from vllm.model_executor.layers.quantization.quark.fused_rotation_mxfp4_quant_moe_sort import (  # noqa: F401
        fused_rotation_mxfp4_quant_moe_sort,
    )
    _has_moe_rot_quant = True
except Exception:
    _has_moe_rot_quant = False

