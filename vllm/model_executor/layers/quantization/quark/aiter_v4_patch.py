#!/usr/bin/env python3
"""
GEAK iter 8: Monkey-patch aiter.fused_moe_2stages to force V4 fused 3-in-1 path.

Replaces:
  Current path A (aiter default for Qwen3-30B):
    hidden_states.matmul(rotation)        ← torch.matmul rotation
    a1 = hidden_states.to(bf16); a1_scale=None  ← stage1 quantizes inline
    stage1(a1=bf16) → bf16 quant inline in MFMA C++
    stage2(...)

With path V4:
    V4(hidden_states, rotation, sorted_ids) → fp4, sorted_scale  ← OUR fused kernel
    a1=fp4, a1_scale=sorted_scale                    ← pre-quantized
    stage1(a1=fp4)        ← skip inline quant (already done)
    stage2(...)

This tests whether the "fused rotation + pre-quantized" path can beat aiter's
"torch.matmul + inline-quant-in-MFMA" optimal default.
"""
import os
import logging
import functools
import torch

logger = logging.getLogger(__name__)

_PATCH_APPLIED = False
_buf_cache: dict = {}


def _get_v4_bufs(M, K, m_pad, m_o, n_i, device, BLOCK_M=32):
    key = (M, K, m_pad, device.index if device.index is not None else 0)
    cached = _buf_cache.get(key)
    if cached is not None:
        return cached
    scratch_rows = ((M + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    total_rows = max(m_pad, m_o) + scratch_rows
    fp4_buf = torch.empty((M, K // 2), dtype=torch.uint8, device=device)
    sc_buf = torch.zeros((total_rows, n_i), dtype=torch.uint8, device=device)
    _buf_cache[key] = (fp4_buf, sc_buf)
    return fp4_buf, sc_buf


def apply_v4_patch():
    """Monkey-patch aiter.fused_moe.fused_moe_2stages to use V4 fused rotation+quant+sort."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    import aiter.fused_moe as afm
    from aiter import QuantType, ActivationType
    from aiter.utility import dtypes
    from vllm.model_executor.layers.quantization.quark.fused_rotation_mxfp4_quant_moe_gluon_v4 import (
        fused_rot_quant_moe_sort_gluon_v4,
        BLOCK_M as V4_BLOCK_M,
    )

    _orig_2stages = afm.fused_moe_2stages

    @functools.wraps(_orig_2stages)
    def _patched_2stages(*args, **kwargs):
        # Extract relevant params
        rotation = kwargs.get('rotation', None)
        rotation_size = kwargs.get('rotation_size', 0)
        quant_type = kwargs.get('quant_type', None)
        activation = kwargs.get('activation', None)

        # Positional args: hidden_states, w1, w2, topk, sorted_ids, sorted_weights,
        #                  sorted_expert_ids, num_valid_ids, moe_out, isG1U1, block_size_M
        hidden_states = args[0]
        w1 = args[1]
        moe_out = args[8]
        dtype = moe_out.dtype

        # Check if V4 path applies (same condition as aiter's first branch)
        is_target = (
            rotation is not None and rotation_size > 0
            and quant_type == QuantType.per_1x32
            and dtype in [dtypes.bf16, dtypes.fp16]
            and w1.dtype == dtypes.fp4x2
            and activation == ActivationType.Swiglu
        )
        if not is_target:
            return _orig_2stages(*args, **kwargs)

        # === V4 FUSED PATH ===
        sorted_ids = args[4]
        sorted_weights = args[5]
        sorted_expert_ids = args[6]
        num_valid_ids = args[7]
        block_size_M = args[10]
        w2 = args[2]
        topk = args[3]
        isG1U1 = args[9]

        # Other kwargs
        doweight_stage1 = kwargs.get('doweight_stage1', False)
        q_dtype_a = kwargs.get('q_dtype_a', None)
        q_dtype_w = kwargs.get('q_dtype_w', None)
        w1_scale = kwargs.get('w1_scale', None)
        w2_scale = kwargs.get('w2_scale', None)
        a2_scale = kwargs.get('a2_scale', None)
        num_local_tokens = kwargs.get('num_local_tokens', None)
        hidden_pad = kwargs.get('hidden_pad', 0)
        intermediate_pad = kwargs.get('intermediate_pad', 0)
        bias1 = kwargs.get('bias1', None)
        bias2 = kwargs.get('bias2', None)

        # === Compute via V4 fused kernel ===
        token_num, K = hidden_states.shape
        E, model_dim, inter_dim = afm.get_inter_dim(w1.shape, w2.shape)
        device = hidden_states.device

        metadata = afm.get_2stage_cfgs(
            afm.get_padded_M(token_num),
            model_dim, inter_dim, E, topk,
            dtype, q_dtype_a, q_dtype_w,
            quant_type, isG1U1, activation, doweight_stage1,
            hidden_pad, intermediate_pad, bias1, bias2,
        )

        # Allocate V4 output buffers (CUDAGraph-stable)
        m_o = sorted_ids.shape[0]
        m_pad = ((m_o + block_size_M - 1) // block_size_M) * block_size_M
        n_i = K // 32
        num_experts = E
        fp4_buf, sc_buf = _get_v4_bufs(token_num, K, m_pad, m_o, n_i, device, V4_BLOCK_M)

        # Call our V4 fused kernel: rotation + quant + sort in ONE
        fp4_a1, sorted_scale = fused_rot_quant_moe_sort_gluon_v4(
            hidden_states, rotation, rotation_size,
            sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
            token_num=token_num, m_pad=m_pad,
            fp4_out=fp4_buf, sorted_scale_out=sc_buf,
            topk=topk, num_experts=num_experts, block_size=block_size_M,
        )

        # Prepare a1, a1_scale for stage1
        a1 = fp4_a1.view(dtypes.fp4x2)
        # sorted_scale is [m_pad, n_i] uint8; aiter expects 5D view (m_pad/32, n_i/8, 4, 16, 4)
        # Our data layout is equivalent (verified via flat index analysis)
        a1_scale = sorted_scale[:m_pad].view(dtypes.fp8_e8m0).reshape(
            m_pad // 32, n_i // 8, 4, 16, 4
        )

        # Allocate a2 output (matching aiter's allocation pattern)
        a2 = torch.empty((token_num, topk, inter_dim), dtype=dtype, device=device)

        # Stage1 with pre-quantized fp4 input
        a2 = metadata.stage1(
            a1, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids,
            a2, topk, block_m=block_size_M,
            a1_scale=a1_scale, w1_scale=w1_scale,
            sorted_weights=sorted_weights if doweight_stage1 else None,
        )

        # Stage2 quantization (replicate aiter's first-branch behavior)
        # For per_1x32 + bf16 + fp4x2 + Swiglu, a2_scale=None and stage2 does inline quant
        a2_scale = None

        # Stage2
        metadata.stage2(
            a2, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids,
            moe_out, topk,
            w2_scale=w2_scale, a2_scale=a2_scale, block_m=block_size_M,
            sorted_weights=sorted_weights if not doweight_stage1 else None,
        )
        return moe_out

    afm.fused_moe_2stages = _patched_2stages
    _PATCH_APPLIED = True
    logger.info("Applied V4 monkey-patch to aiter.fused_moe.fused_moe_2stages (V4 fused 3-in-1 path)")
