"""
Fused Rotation + MXFP4 Quantization integration for MoE experts in vLLM.

Replaces the separated path:
  x → torch.matmul(rotation) → aiter.fused_moe(bf16→internal_quant+GEMM)

With:
  x → fused_rot_quant(bf16→fp4+scales) → sort_scales → ck_moe_gemm(fp4→GEMM)

The fused kernel eliminates the bf16 intermediate buffer between rotation
and quantization, reducing global memory traffic by ~4x for the activation
data.

Environment variable:
  VLLM_MOE_FUSED_ROTATION=1  — enable fused rotation+quant for MoE
"""

import os
import logging

import torch

logger = logging.getLogger(__name__)

FUSED_MOE_ROTATION = os.getenv("VLLM_MOE_FUSED_ROTATION", "0") == "1"


def fused_moe_with_rotation(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    rotation: torch.Tensor,
    rotation_size: int,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    expert_mask: torch.Tensor | None = None,
    activation_str: str = "silu",
) -> torch.Tensor:
    """
    Full MoE pipeline with fused rotation+quantization for stage1.

    Stage1: fused_rotation_mxfp4_quant → sort_scales → ck_moe GEMM (fp4 input)
    Stage2: cktile_moe_stage2 (bf16 intermediate, internal quant+GEMM)

    The fp4 data stays in original token order (ck_moe uses sorted_token_ids
    for indirection). Scales are sorted by moe_mxfp4_sort to match the GEMM
    kernel's expectations.
    """
    import aiter
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import moe_sorting, get_inter_dim
    from aiter.utility import fp4_utils

    from vllm.model_executor.layers.quantization.quark.fused_rotation_mxfp4_quant_moe import (
        fused_rotation_mxfp4_quant_moe,
    )

    M, topk = topk_ids.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    isG1U1 = inter_dim != w1.shape[1]

    global_E = E
    if expert_mask is not None:
        global_E = expert_mask.numel()

    activation = (
        ActivationType.Silu if activation_str == "silu" else ActivationType.Gelu
    )

    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    block_size_M = 32 if M < 16384 else 64

    # Step 1: MoE sorting (same as aiter's normal path)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = (
        moe_sorting(
            topk_ids,
            topk_weights,
            global_E,
            model_dim,
            dtype,
            block_size_M,
            expert_mask,
        )
    )
    token_num = sorted_ids.shape[0]

    # Step 2: Fused rotation + MXFP4 quantization (our SW FP4 Triton kernel)
    # Output: fp4 in original token order, scales in original token order
    a1_fp4_uint8, a1_scale_uint8 = fused_rotation_mxfp4_quant_moe(
        hidden_states, rotation, rotation_size
    )

    # Step 3: Convert fp4 uint8 → fp4x2 dtype (for aiter GEMM kernel)
    fp4x2_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4x2_dtype is not None:
        a1 = a1_fp4_uint8.view(fp4x2_dtype)
    else:
        a1 = a1_fp4_uint8

    # Step 4: Sort scales into the tiled 5D format expected by GEMM kernels
    a1_scale = fp4_utils.moe_mxfp4_sort(
        a1_scale_uint8,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=M,
        block_size=block_size_M,
    )

    # Step 5: Stage1 GEMM — ck_moe with fp4 input + sorted scales
    D = inter_dim
    a2 = torch.empty(
        (token_num, topk, D), dtype=dtype, device=device
    )

    w1_scale_typed = (
        w1_scale.view(dtypes.fp8_e8m0)
        if w1.dtype == dtypes.fp4x2
        else w1_scale
    )

    aiter.ck_moe_stage1_fwd(
        a1,                  # fp4x2 input (original token order)
        w1,                  # [E, 2*inter_dim, model_dim//2] fp4x2 weights
        w2,                  # [E, model_dim, inter_dim//2]
        sorted_ids,          # indices into a1
        sorted_expert_ids,   # expert per block
        num_valid_ids,       # valid count
        a2,                  # output [token_num, topk, D]
        topk,
        "",                  # kernelName (auto-select)
        w1_scale_typed,      # weight scales
        a1_scale,            # activation scales (sorted 5D)
        block_size_M,
        None,                # sorted_weights (not used for stage1)
        QuantType.per_1x32,  # quantization type
        activation,          # Swiglu
    )

    # Step 6: Stage2 GEMM — cktile handles quant internally
    w2_scale_typed = (
        w2_scale.view(dtypes.fp8_e8m0)
        if w2.dtype == dtypes.fp4x2
        else w2_scale
    )

    aiter.moe_cktile2stages_gemm2(
        a2,                  # XQ: intermediate [token_num, topk, D] bf16
        w2,                  # WQ: [E, model_dim, inter_dim//2] fp4x2
        moe_buf,             # Y: output [M, model_dim]
        sorted_ids,          # sorted_ids
        sorted_expert_ids,   # sorted_expert_ids
        num_valid_ids,       # max_token_ids
        topk,                # topk
        0,                   # n_padded_zeros
        0,                   # k_padded_zeros
        sorted_weights,      # topk_weight (routing weights)
        None,                # x_scale (a2_scale, None = internal quant)
        w2_scale_typed,      # w_scale
        None,                # exp_bias
        block_size_M,        # block_m
    )

    return moe_buf
