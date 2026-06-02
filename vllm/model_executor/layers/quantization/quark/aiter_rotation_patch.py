#!/usr/bin/env python3
"""
MoE Gluon rotation patch — replaces torch.matmul rotation with Gluon kernel.

Insight: Gluon rotation-only kernel (no quant, no scatter) beats torch.matmul
for M=4-16 (where most decode batches land). Saves 0-3µs per MoE call.

Strategy:
- Replace ONLY the torch.matmul rotation step in aiter's fused_moe_2stages
- Keep aiter's first branch (bf16 → asm_stage1 with inline quant) - optimal
- Best of both worlds: faster rotation + optimal inline quant

Activation (in priority order):
  VLLM_FUSED_ROTATION=1           → unified switch (also enables Dense Gluon)
  VLLM_MOE_FORCE_GLUON_ROTATION=1 → legacy MoE-only switch

Both are handled in fused_moe_rotation.py which calls apply_gluon_rotation_patch().
"""
import os
import logging
import torch

logger = logging.getLogger(__name__)
_PATCH_APPLIED = False


def apply_gluon_rotation_patch():
    """Monkey-patch aiter to use Gluon rotation instead of torch.matmul."""
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
        rotation = kwargs.get('rotation', None)
        rotation_size = kwargs.get('rotation_size', 0)

        # Only replace rotation if it's provided
        if rotation is None or rotation_size <= 0:
            return _orig_2stages(*args, **kwargs)

        # Pre-compute rotated hidden_states using Gluon kernel
        hidden_states = args[0]
        if hidden_states.dim() == 2 and hidden_states.dtype == torch.bfloat16:
            M = hidden_states.shape[0]
            # Adaptive dispatch: V3 (k_width=8 + num_stages=3) for M>=4,
            #                    V2 (k_width=4) for M<4 (V3 has slight regression at M=1)
            # V3 wins by 0.8-2.2us at M=4..16 vs V2 / torch.matmul
            if M >= 4:
                rotated = gluon_rotation_v3(hidden_states, rotation, rotation_size)
            else:
                rotated = gluon_rotation_v2(hidden_states, rotation, rotation_size,
                                             persistent=False)
            # Pass rotated hidden_states to aiter with rotation=None (already done)
            new_args = (rotated,) + args[1:]
            new_kwargs = dict(kwargs)
            new_kwargs['rotation'] = None
            new_kwargs['rotation_size'] = 0
            return _orig_2stages(*new_args, **new_kwargs)
        else:
            # Fallback to original path
            return _orig_2stages(*args, **kwargs)

    afm.fused_moe_2stages = _patched_2stages
    _PATCH_APPLIED = True
    logger.info("Applied Gluon rotation monkey-patch (replaces torch.matmul in aiter)")
