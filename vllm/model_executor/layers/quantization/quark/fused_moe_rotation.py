"""
MoE rotation feature flags and kernel activation.

Environment variables:
  VLLM_FUSED_ROTATION=1   → enable both Dense Gluon and MoE Gluon kernels (recommended)
  VLLM_FUSED_ROTATION=0   → disable both
  VLLM_MOE_FUSED_ROTATION=0          → disable MoE rotation pipeline entirely
  VLLM_MOE_FORCE_GLUON_ROTATION=1    → enable MoE Gluon only (legacy; use VLLM_FUSED_ROTATION instead)
"""

import os

# Whether MoE fused rotation+quant is enabled (aiter 3-in-1 kernel path).
FUSED_MOE_ROTATION: bool = os.getenv("VLLM_MOE_FUSED_ROTATION", "1") == "1"


def _is_moe_gluon_rotation_enabled() -> bool:
    """Return True if MoE Gluon rotation patch should be applied.

    Priority:
      VLLM_FUSED_ROTATION=1            → True  (unified switch, recommended)
      VLLM_FUSED_ROTATION=0            → False
      VLLM_MOE_FORCE_GLUON_ROTATION=1  → True  (legacy MoE-only switch)
    """
    unified = os.environ.get("VLLM_FUSED_ROTATION", "").strip().lower()
    if unified in ("1", "true"):
        return True
    if unified in ("0", "false"):
        return False
    return os.getenv("VLLM_MOE_FORCE_GLUON_ROTATION", "0") == "1"


# Apply MoE Gluon monkey-patch eagerly if enabled so aiter sees the patched
# fused_moe_2stages before any MoE forward pass.
if _is_moe_gluon_rotation_enabled():
    try:
        from vllm.model_executor.layers.quantization.quark.aiter_rotation_patch import (
            apply_gluon_rotation_patch,
        )
        apply_gluon_rotation_patch()
    except Exception as e:
        import logging
        logging.warning("Failed to apply MoE Gluon rotation patch: %s", e)

try:
    from vllm.model_executor.layers.quantization.quark.fused_rotation_mxfp4_quant_moe_sort import (  # noqa: F401
        fused_rotation_mxfp4_quant_moe_sort,
    )
    _has_moe_rot_quant = True
except Exception:
    _has_moe_rot_quant = False
