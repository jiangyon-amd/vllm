"""
Fused Rotation + MXFP4 Quantization for MoE experts.

Replaces the separated path:
  x → torch.matmul(rotation) → fused_dynamic_mxfp4_quant_moe_sort()

With a single Triton kernel:
  x → [load + rotation matmul + mxfp4 quant + fp4 store] (one kernel)

Then the existing moe_sort phase handles scale sorting.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rotation_mxfp4_quant_kernel(
    x_ptr,
    rot_ptr,
    x_fp4_ptr,
    scale_ptr,
    M, N,
    stride_x_m, stride_x_n,
    stride_rot_r, stride_rot_c,
    stride_fp4_m,
    stride_sc_m,
    RS: tl.constexpr,
    QG: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    Fused rotation + MXFP4 quantization kernel for MoE.
    Each program processes [BLOCK_M, RS] block: rotation_size columns.
    RS must equal rotation_size (128).
    QG = MXFP4 quant group size (32). RS = 4 * QG.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)  # which rotation block (K // RS)

    m_base = pid_m * BLOCK_M
    col_base = pid_k * RS

    # Load x [BLOCK_M, RS]
    offs_m = m_base + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, RS)
    x_offs = offs_m[:, None] * stride_x_m + (col_base + offs_k)[None, :] * stride_x_n
    m_mask = offs_m < M
    x = tl.load(x_ptr + x_offs, mask=m_mask[:, None], other=0.0).to(tl.float32)

    # Load rotation [RS, RS]
    offs_r = tl.arange(0, RS)
    offs_c = tl.arange(0, RS)
    rot = tl.load(rot_ptr + offs_r[:, None] * stride_rot_r + offs_c[None, :] * stride_rot_c).to(tl.float32)

    # Rotation matmul: [BLOCK_M, RS] @ [RS, RS] → [BLOCK_M, RS]
    acc = tl.dot(x.to(tl.bfloat16), rot.to(tl.bfloat16))
    # Truncate to bf16 precision to match torch matmul behavior
    acc = acc.to(tl.bfloat16).to(tl.float32)

    # MXFP4 quantization per QG=32 group
    NUM_QG: tl.constexpr = RS // QG  # 4

    acc_g = tl.reshape(acc, (BLOCK_M, NUM_QG, QG))

    # Scale calculation (per group amax)
    amax = tl.max(tl.abs(acc_g), axis=-1)  # [BLOCK_M, NUM_QG]

    # Scale calculation — match aiter's rounding (0x200000)
    amax_i32 = amax.to(tl.int32, bitcast=True)
    amax_rounded = (amax_i32 + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax_f32 = amax_rounded.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax_f32).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
    e8m0_u8 = (scale_e8m0_unbiased.to(tl.uint8) + 127)

    quant_scale = tl.exp2(-scale_e8m0_unbiased)

    # Quantize: scale activations
    quant_scale_full = tl.reshape(
        tl.broadcast_to(tl.reshape(quant_scale, (BLOCK_M, NUM_QG, 1)), (BLOCK_M, NUM_QG, QG)),
        (BLOCK_M, RS)
    )
    qx = acc * quant_scale_full
    qx = qx.to(tl.uint32, bitcast=True)

    # FP32 → FP4 conversion (match aiter's software implementation)
    s = qx & 0x80000000
    e = (qx >> 23) & 0xFF
    m = qx & 0x7FFFFF

    E8_BIAS: tl.constexpr = 127
    E2_BIAS: tl.constexpr = 1

    adjusted_exponents = tl.core.sub(E8_BIAS, e + 1, sanitize_overflow=False)
    m = tl.where(e < E8_BIAS, (0x400000 | (m >> 1)) >> adjusted_exponents, m)
    e = tl.maximum(e, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

    e2m1_tmp = tl.minimum((((e << 2) | (m >> 21)) + 1) >> 1, 0x7)
    e2m1_value = ((s >> 28) | e2m1_tmp).to(tl.uint8)

    # Pack FP4 pairs
    e2m1_value = tl.reshape(e2m1_value, (BLOCK_M, RS // 2, 2))
    evens, odds = tl.split(e2m1_value)
    packed = evens | (odds << 4)

    # Store FP4
    fp4_base = pid_k * (RS // 2)
    fp4_offs_m = m_base + tl.arange(0, BLOCK_M)
    fp4_offs_n = fp4_base + tl.arange(0, RS // 2)
    fp4_offs = fp4_offs_m[:, None] * stride_fp4_m + fp4_offs_n[None, :]
    tl.store(x_fp4_ptr + fp4_offs, packed, mask=m_mask[:, None])

    # Store scales (raw, not shuffled)
    sc_base = pid_k * NUM_QG
    sc_offs_m = m_base + tl.arange(0, BLOCK_M)
    sc_offs_n = sc_base + tl.arange(0, NUM_QG)
    sc_offs = sc_offs_m[:, None] * stride_sc_m + sc_offs_n[None, :]
    tl.store(scale_ptr + sc_offs, e8m0_u8, mask=m_mask[:, None])


def fused_rotation_mxfp4_quant_moe(
    x: torch.Tensor,
    rotation: torch.Tensor,
    rotation_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused rotation + MXFP4 quantization for MoE expert inputs.

    Args:
        x: [M, K] bf16 hidden states
        rotation: [RS, RS] rotation matrix
        rotation_size: rotation block size (default 128)

    Returns:
        (fp4_out, scales_out): quantized output and scales
    """
    assert x.ndim == 2
    M, K = x.shape
    assert K % rotation_size == 0
    RS = rotation_size
    QG = 32
    n_scales = K // QG

    fp4_out = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
    scales_out = torch.empty((M, n_scales), dtype=torch.uint8, device=x.device)

    BLOCK_M = min(128, max(1, triton.next_power_of_2(M)))
    grid = (triton.cdiv(M, BLOCK_M), K // RS)

    _fused_rotation_mxfp4_quant_kernel[grid](
        x, rotation, fp4_out, scales_out,
        M, K,
        x.stride(0), x.stride(1),
        rotation.stride(0), rotation.stride(1),
        fp4_out.stride(0),
        scales_out.stride(0),
        RS=RS, QG=QG, BLOCK_M=BLOCK_M,
    )

    return fp4_out, scales_out
