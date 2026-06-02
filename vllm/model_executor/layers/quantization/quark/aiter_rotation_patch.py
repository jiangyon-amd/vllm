"""
MoE Gluon rotation patch — replaces torch.matmul rotation in aiter with Gluon kernels.

aiter's fused_moe_2stages computes rotation via:
    hidden_states.reshape(M, K // RS, RS).matmul(rotation).reshape(M, K)

This patch replaces that step with a Gluon Triton kernel (no quant, no scatter)
that is faster for decode batch sizes (M=1-32):
  - gluon_rotation_v3 (k_width=8, 3 pipeline stages): optimal for M≥4 (+0.8-2.2µs)
  - gluon_rotation_v2 (k_width=4):                    fallback for M<4

The rest of fused_moe_2stages (inline MXFP4 quant via asm_stage1) is preserved.
Activation: import this module via fused_moe_rotation.py (VLLM_FUSED_ROTATION=1).
"""
import logging
import torch

logger = logging.getLogger(__name__)
_PATCH_APPLIED = False


def apply_gluon_rotation_patch() -> None:
    """Monkey-patch aiter.fused_moe_2stages to use Gluon rotation kernels."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    import aiter.fused_moe as afm
    from vllm.model_executor.layers.quantization.quark.gluon_rotation_only_v2 import (
        gluon_rotation_v2,
    )
    from vllm.model_executor.layers.quantization.quark.gluon_rotation_only_v3 import (
        gluon_rotation_v3,
    )

    _orig_2stages = afm.fused_moe_2stages

    def _patched_2stages(*args, **kwargs):
        rotation = kwargs.get("rotation")
        rotation_size = kwargs.get("rotation_size", 0)

        if rotation is None or rotation_size <= 0:
            return _orig_2stages(*args, **kwargs)

        hidden_states = args[0]
        if hidden_states.dim() == 2 and hidden_states.dtype == torch.bfloat16:
            M = hidden_states.shape[0]
            # V3 (k_width=8 + 3-stage pipeline) wins for M≥4; V2 for M<4.
            if M >= 4:
                rotated = gluon_rotation_v3(hidden_states, rotation, rotation_size)
            else:
                rotated = gluon_rotation_v2(hidden_states, rotation, rotation_size,
                                            persistent=False)
            new_kwargs = {**kwargs, "rotation": None, "rotation_size": 0}
            return _orig_2stages(*((rotated,) + args[1:]), **new_kwargs)
        return _orig_2stages(*args, **kwargs)

    afm.fused_moe_2stages = _patched_2stages
    _PATCH_APPLIED = True
    logger.info("Applied Gluon rotation patch to aiter.fused_moe_2stages")
