"""
MoE Gluon rotation patch — replaces torch.matmul rotation in aiter with Gluon kernel.

aiter's fused_moe_2stages computes rotation via:
    hidden_states.reshape(M, K // RS, RS).matmul(rotation).reshape(M, K)

This patch replaces that step with a Gluon MFMA kernel (faster for decode M=1-32):
  k_width=8 (M≥4): +0.8-2.2µs over torch.matmul
  k_width=4 (M<4):  fallback for tiny batch

Activation: VLLM_FUSED_ROTATION=1 (via fused_moe_rotation.py).
"""
import logging
import torch

logger = logging.getLogger(__name__)
_PATCH_APPLIED = False


def apply_gluon_rotation_patch() -> None:
    """Monkey-patch aiter.fused_moe_2stages to use Gluon rotation kernel."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    import aiter.fused_moe as afm
    from vllm.model_executor.layers.quantization.quark.gluon_rotation_moe import (
        gluon_moe_rotation,
    )

    _orig_2stages = afm.fused_moe_2stages

    def _patched_2stages(*args, **kwargs):
        rotation = kwargs.get("rotation")
        rotation_size = kwargs.get("rotation_size", 0)
        if rotation is None or rotation_size <= 0:
            return _orig_2stages(*args, **kwargs)

        hidden_states = args[0]
        if hidden_states.dim() == 2 and hidden_states.dtype == torch.bfloat16:
            rotated = gluon_moe_rotation(hidden_states, rotation, rotation_size)
            new_kwargs = {**kwargs, "rotation": None, "rotation_size": 0}
            return _orig_2stages(*((rotated,) + args[1:]), **new_kwargs)
        return _orig_2stages(*args, **kwargs)

    afm.fused_moe_2stages = _patched_2stages
    _PATCH_APPLIED = True
    logger.info("Applied Gluon rotation patch to aiter.fused_moe_2stages")
