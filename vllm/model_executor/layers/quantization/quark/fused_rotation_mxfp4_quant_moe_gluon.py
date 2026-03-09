"""
Fused Rotation + MXFP4 Quantization for MoE experts — Gluon version.

Uses AMD Gluon (MFMA + buffer_load/store + shared memory) for rotation matmul,
with aiter-compatible scale rounding (0x200000) and software FP4 conversion.
"""

import torch
import triton
from triton import language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _fused_rot_quant_moe_gluon(
    x_ptr, rot_ptr, fp4_ptr, scale_ptr,
    M, N,
    stride_x_m, stride_rot_r, stride_rot_c,
    stride_fp4_m, stride_sc_m,
    RS: gl.constexpr,
    QG: gl.constexpr,
    BLOCK_M: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    pid_m = gl.program_id(0)
    pid_rot = gl.program_id(1)

    # ======== Layouts ========
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16], transposed=True,
        warps_per_cta=[2, 2], tiles_per_warp=[1, 4],
    )
    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4], threads_per_warp=[16, 4],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0],
    )
    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[4, 1], threads_per_warp=[4, 16],
        warps_per_cta=[1, NUM_WARPS], order=[0, 1],
    )
    shared_rot: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[1, 0],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=4)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=4)

    NUM_QG: gl.constexpr = RS // QG
    HALF_QG: gl.constexpr = QG // 2

    # ======== Offsets ========
    m_base = pid_m * BLOCK_M
    col_base = pid_rot * RS
    offs_m = m_base + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked_mk))
    m_mask = offs_m < M

    # ======== Load x [BLOCK_M, RS] via buffer_load ========
    offs_k = gl.arange(0, RS, layout=gl.SliceLayout(0, blocked_mk))
    x_offs = offs_m[:, None] * stride_x_m + (col_base + offs_k)[None, :]
    x_tile = gl.amd.cdna4.buffer_load(ptr=x_ptr, offsets=x_offs, mask=m_mask[:, None])

    # ======== Load rotation [RS, RS] to shared memory ========
    smem_rot = gl.allocate_shared_memory(gl.bfloat16, [RS, RS], layout=shared_rot)
    offs_rr = gl.arange(0, RS, layout=gl.SliceLayout(1, blocked_kn))
    offs_rc = gl.arange(0, RS, layout=gl.SliceLayout(0, blocked_kn))
    rot_data = gl.amd.cdna4.buffer_load(ptr=rot_ptr,
        offsets=offs_rr[:, None] * stride_rot_r + offs_rc[None, :] * stride_rot_c)
    smem_rot.store(rot_data)

    # ======== MFMA rotation matmul ========
    acc = gl.zeros((BLOCK_M, RS), gl.float32, layout=mfma_layout)
    acc = gl.amd.cdna4.mfma(gl.convert_layout(x_tile, dot_a),
                             smem_rot.load(layout=dot_b), acc)
    acc = acc.to(gl.bfloat16).to(gl.float32)

    # ======== Quantization — aiter-compatible (0x200000 rounding) ========
    acc_g = gl.reshape(acc, (BLOCK_M, NUM_QG, QG))
    amax = gl.max(gl.abs(acc_g), axis=-1)

    # aiter-style scale: 0x200000 rounding (round-to-nearest)
    amax_i32 = amax.to(gl.int32, bitcast=True)
    amax_rounded = (amax_i32 + 0x200000).to(gl.uint32, bitcast=True) & 0xFF800000
    amax_f32 = amax_rounded.to(gl.float32, bitcast=True)
    scale_unbiased = gl.log2(amax_f32)
    # floor via int truncation
    scale_unbiased = scale_unbiased.to(gl.int32).to(gl.float32) - 2.0
    scale_unbiased = gl.maximum(gl.minimum(scale_unbiased, 127.0), -127.0)
    e8m0_u8 = (scale_unbiased.to(gl.int32) + 127).to(gl.uint8)

    # ======== Software FP4 conversion (aiter-compatible, 100% match) ========
    quant_scale = gl.exp2(-scale_unbiased)
    quant_scale_full = acc_g * 0.0 + gl.expand_dims(quant_scale, axis=2)
    qx = acc_g * quant_scale_full
    qx_flat = gl.reshape(qx, (BLOCK_M, RS))

    qx_u32 = qx_flat.to(gl.uint32, bitcast=True)
    s = qx_u32 & 0x80000000
    e = (qx_u32 >> 23) & 0xFF
    m = qx_u32 & 0x7FFFFF

    E8_BIAS: gl.constexpr = 127
    E2_BIAS: gl.constexpr = 1

    adjusted_exp = E8_BIAS - e - 1
    m_denorm = (0x400000 | (m >> 1)) >> adjusted_exp
    m = gl.where(e < E8_BIAS, m_denorm, m)
    e = gl.maximum(e, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

    e2m1_tmp = gl.minimum((((e << 2) | (m >> 21)) + 1) >> 1, 7)
    e2m1_value = ((s >> 28) | e2m1_tmp).to(gl.uint8)

    e2m1_value = gl.reshape(e2m1_value, (BLOCK_M, RS // 2, 2))
    evens, odds = gl.split(e2m1_value)
    packed = evens | (odds << 4)

    # ======== Store FP4 via buffer_store ========
    fp4_store: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4], threads_per_warp=[2, 32],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0],
    )
    packed_s = gl.convert_layout(packed, fp4_store)
    fp4_base = pid_rot * (RS // 2)
    s_m = m_base + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, fp4_store))
    s_mask = s_m < M
    fp4_c = fp4_base + gl.arange(0, RS // 2, layout=gl.SliceLayout(0, fp4_store))
    gl.amd.cdna4.buffer_store(stored_value=packed_s, ptr=fp4_ptr,
        offsets=s_m[:, None] * stride_fp4_m + fp4_c[None, :], mask=s_mask[:, None])

    # ======== Store scales (raw, no shuffle) via tl.store ========
    sc_store: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1], threads_per_warp=[32, 2],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0],
    )
    sc_s = gl.convert_layout(e8m0_u8, sc_store)
    sc_m = m_base + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, sc_store))
    sc_mask = sc_m < M
    sc_base = pid_rot * NUM_QG
    sc_c = sc_base + gl.arange(0, NUM_QG, layout=gl.SliceLayout(0, sc_store))
    tl.store(scale_ptr + sc_m[:, None] * stride_sc_m + sc_c[None, :], sc_s, mask=sc_mask[:, None])


def fused_rotation_mxfp4_quant_moe_gluon(
    x: torch.Tensor,
    rotation: torch.Tensor,
    rotation_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gluon-based fused rotation + MXFP4 quantization for MoE.
    Uses MFMA for rotation matmul, aiter-compatible scale rounding.
    """
    assert x.ndim == 2
    M, K = x.shape
    assert K % rotation_size == 0
    RS = rotation_size
    QG = 32
    n_scales = K // QG

    fp4_out = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
    scales_out = torch.empty((M, n_scales), dtype=torch.uint8, device=x.device)

    BLOCK_M = 32
    NUM_WARPS = 4
    grid = (triton.cdiv(M, BLOCK_M), K // RS)

    _fused_rot_quant_moe_gluon[grid](
        x, rotation, fp4_out, scales_out,
        M, K,
        x.stride(0), rotation.stride(0), rotation.stride(1),
        fp4_out.stride(0), scales_out.stride(0),
        RS=RS, QG=QG, BLOCK_M=BLOCK_M,
        NUM_WARPS=NUM_WARPS, num_warps=NUM_WARPS,
    )

    return fp4_out, scales_out
