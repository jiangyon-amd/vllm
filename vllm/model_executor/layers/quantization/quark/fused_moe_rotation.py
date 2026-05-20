"""
Compatibility shim for MoE fused-rotation feature flags.

The original module provided fused MoE rotation pipeline wiring. In this
workspace, the active implementation lives in
`fused_rotation_mxfp4_quant_moe_sort.py`. Qwen3 MoE only imports two flags
from this module to decide whether to pass rotation into the ROCm fused MoE
path, so we keep this lightweight compatibility layer.
"""

import os

FUSED_MOE_ROTATION = os.getenv("VLLM_MOE_FUSED_ROTATION", "1") == "1"

# Default to HIP MFMA kernel (4x faster than Gluon Triton kernels).
# User can override with AITER_MOE_HIP_MFMA=0 to fall back to Triton.
os.environ.setdefault("AITER_MOE_HIP_MFMA", "1")

# GEAK iter 8: Apply V4 monkey-patch if enabled (forces real fused 3-in-1 path)
if os.getenv("VLLM_MOE_FORCE_FUSED_V4", "0") == "1":
    try:
        from vllm.model_executor.layers.quantization.quark.aiter_v4_patch import (
            apply_v4_patch,
        )
        apply_v4_patch()
    except Exception as e:
        import logging
        logging.warning(f"Failed to apply V4 monkey-patch: {e}")

# GEAK iter 9: Apply Gluon rotation patch (replaces torch.matmul with Gluon kernel)
if os.getenv("VLLM_MOE_FORCE_GLUON_ROTATION", "0") == "1":
    try:
        from vllm.model_executor.layers.quantization.quark.aiter_rotation_patch import (
            apply_gluon_rotation_patch,
        )
        apply_gluon_rotation_patch()
    except Exception as e:
        import logging
        logging.warning(f"Failed to apply Gluon rotation patch: {e}")

try:
    # Validate that the fused MoE rotation implementation is importable.
    from vllm.model_executor.layers.quantization.quark.fused_rotation_mxfp4_quant_moe_sort import (  # noqa: F401
        fused_rotation_mxfp4_quant_moe_sort,
    )
    _has_moe_rot_quant = True
except Exception:
    _has_moe_rot_quant = False

