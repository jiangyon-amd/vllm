"""
Compatibility shim for MoE fused-rotation feature flags.

The original module provided fused MoE rotation pipeline wiring. In this
workspace, the active implementation lives in
`fused_rotation_mxfp4_quant_moe_sort.py`. Qwen3 MoE only imports two flags
from this module to decide whether to pass rotation into the ROCm fused MoE
path, so we keep this lightweight compatibility layer.
"""

import os

FUSED_MOE_ROTATION = os.getenv("VLLM_MOE_FUSED_ROTATION", "0") == "1"

try:
    # Validate that the fused MoE rotation implementation is importable.
    from vllm.model_executor.layers.quantization.quark.fused_rotation_mxfp4_quant_moe_sort import (  # noqa: F401
        fused_rotation_mxfp4_quant_moe_sort,
    )
    _has_moe_rot_quant = True
except Exception:
    _has_moe_rot_quant = False

