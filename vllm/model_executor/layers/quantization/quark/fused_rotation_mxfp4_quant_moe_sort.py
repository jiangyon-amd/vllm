"""
Fused Rotation + MXFP4 Quant + MoE Sort.

Replaces: torch.matmul(rotation) + fused_dynamic_mxfp4_quant_moe_sort  [2 kernels + bf16 GMEM]
With:     Gluon rotation+quant + moe_mxfp4_sort                        [2 kernels, no bf16 GMEM]

Same kernel count, saves bf16 intermediate GMEM read/write.
"""

import os
import logging
import torch
import triton
import triton.language as tl
from aiter.utility import dtypes
from aiter.utility.fp4_utils import moe_mxfp4_sort
from vllm.model_executor.layers.quantization.quark.fused_rotation_quant_gluon_v2 import (
    fused_gluon_v2,
)
from vllm.model_executor.layers.quantization.quark.fused_rotation_quant_gluon_v2_kw8 import (
    fused_gluon_v2_kw8,
)

# ---------------------------------------------------------------------------
# Register fused_rotation_mxfp4_quant_moe_sort as a custom op so that
# CUDAGraph can capture it and torch.compile treats it as opaque.
# ---------------------------------------------------------------------------
_has_moe_rot_quant_custom_op = False
try:
    from vllm.utils.torch_utils import direct_register_custom_op
    from vllm.platforms import current_platform

    def _fused_rot_quant_moe_sort_op(
        x: torch.Tensor,
        rotation: torch.Tensor,
        rotation_size: int,
        sorted_ids: torch.Tensor,
        num_valid_ids: torch.Tensor,
        token_num: int,
        topk: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Direct kernel call — no Python dispatch overhead.
        return _fused_rot_quant_moe_sort_impl(
            x, rotation, rotation_size,
            sorted_ids, num_valid_ids,
            token_num, topk, block_size,
        )

    def _fused_rot_quant_moe_sort_fake(
        x: torch.Tensor,
        rotation: torch.Tensor,
        rotation_size: int,
        sorted_ids: torch.Tensor,
        num_valid_ids: torch.Tensor,
        token_num: int,
        topk: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        M, K = x.shape
        QG = 32
        n_i = K // QG
        m_o = sorted_ids.shape[0]
        m_pad = ((m_o + 31) // 32) * 32
        return (
            torch.empty((M, K // 2), dtype=dtypes.fp4x2, device=x.device),
            torch.empty((m_pad, n_i), dtype=dtypes.fp8_e8m0, device=x.device),
        )

    direct_register_custom_op(
        op_name="fused_rotation_mxfp4_quant_moe_sort",
        op_func=_fused_rot_quant_moe_sort_op,
        mutates_args=[],
        fake_impl=_fused_rot_quant_moe_sort_fake,
        dispatch_key=current_platform.dispatch_key,
    )
    _has_moe_rot_quant_custom_op = True
    print("[INFO] Registered fused_rotation_mxfp4_quant_moe_sort as custom op (CUDAGraph compatible)")
except Exception as e:
    logging.getLogger(__name__).warning(
        "Failed to register fused_rotation_mxfp4_quant_moe_sort custom op: %s", e
    )

ENABLE_GLUON_SORTED_SCALE_FUSION = (
    os.getenv("VLLM_MOE_GLUON_SORTED_SCALE_FUSION", "0") == "1"
)
ENABLE_HIP_FUSED_ROTATION = (
    os.getenv("VLLM_MOE_HIP_FUSED_ROTATION", "0") == "1"
)
ENABLE_HIP_MFMA = (
    os.getenv("VLLM_MOE_HIP_MFMA", "0") == "1"
)
ENABLE_TRITON_ROT_SORT_FUSION = (
    os.getenv("VLLM_MOE_TRITON_ROT_SORT_FUSION", "0") == "1"
)
ENABLE_GLUON_MOE_DECODE = (
    os.getenv("VLLM_MOE_GLUON_DECODE", "1") == "1"
)
ENABLE_GLUON_KW8 = (
    os.getenv("VLLM_MOE_GLUON_KW8", "1") == "1"
)
logger = logging.getLogger(__name__)
_DISPATCH_LOG_KEYS: set[tuple] = set()


def _pick_decode_max_q(m_pad: int) -> int:
    if m_pad <= 64:
        return 64
    elif m_pad <= 128:
        return 128
    else:
        return 256


@triton.jit
def _fused_decode_m1_rot_quant_sorted_kernel(
    x_ptr,
    rot_ptr,
    fp4_ptr,
    sorted_ids_ptr,
    num_valid_ids_ptr,
    sorted_scale_ptr,
    stride_x_m,
    stride_x_n,
    stride_rot_r,
    stride_rot_c,
    stride_fp4_m,
    stride_sc_m,
    stride_sc_n,
    token_num,
    n_i,
    tile_n,
    m_o,
    RS: tl.constexpr,
    QG: tl.constexpr,
    MAX_Q: tl.constexpr,
):
    # One program per rotation block.
    pid_k = tl.program_id(0)
    col_base = pid_k * (RS // QG)
    NUM_QG: tl.constexpr = RS // QG

    # Load x[0, col_base*QG : ...] and rot[RS, RS], then rotate.
    offs_k = tl.arange(0, RS)
    x = tl.load(x_ptr + 0 * stride_x_m + (pid_k * RS + offs_k) * stride_x_n).to(tl.float32)
    offs_r = tl.arange(0, RS)
    offs_c = tl.arange(0, RS)
    rot = tl.load(
        rot_ptr + offs_r[:, None] * stride_rot_r + offs_c[None, :] * stride_rot_c
    ).to(tl.float32)
    acc = tl.dot(x[None, :].to(tl.bfloat16), rot.to(tl.bfloat16))
    acc = acc.to(tl.bfloat16).to(tl.float32)

    # MXFP4 quantization using v_cvt ISA (replaces ~15 lines of manual bit ops).
    HALF_QG: tl.constexpr = QG // 2
    acc_g = tl.reshape(acc, (1, NUM_QG, QG))
    amax = tl.max(tl.abs(acc_g), axis=-1, keep_dims=False)

    # Direct exponent extraction (replaces log2/exp2/floor with simple bit ops)
    amax_u32 = amax.to(tl.uint32, bitcast=True)
    amax_u32 = (amax_u32 + 0x200000) & 0xFF800000
    raw_exp = (amax_u32 >> 23) & 0xFF
    e8m0 = tl.maximum(raw_exp, 2) - 2
    e8m0_u8 = e8m0.to(tl.uint8)

    # hw_scale for v_cvt (divides input by this value internally)
    hw_scale = (e8m0.to(tl.uint32) << 23).to(tl.float32, bitcast=True)

    # Reshape to even/odd pairs for v_cvt_scalef32_pk_fp4_f32
    acc_pairs = tl.reshape(acc_g, (1, NUM_QG, HALF_QG, 2))
    even_vals, odd_vals = tl.split(acc_pairs)
    even_flat = tl.reshape(even_vals, (1, NUM_QG * HALF_QG))
    odd_flat = tl.reshape(odd_vals, (1, NUM_QG * HALF_QG))
    hw_scale_broad = tl.broadcast_to(
        tl.reshape(hw_scale, (1, NUM_QG, 1)), (1, NUM_QG, HALF_QG)
    )
    scale_flat = tl.reshape(hw_scale_broad, (1, NUM_QG * HALF_QG))

    packed_u32 = tl.inline_asm_elementwise(
        asm="v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3",
        constraints="=v,v,v,v",
        args=[even_flat, odd_flat, scale_flat],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    packed = packed_u32.to(tl.uint8)
    packed = tl.reshape(packed, (RS // 2,))

    # Store fp4 for row 0.
    fp4_offs_n = pid_k * (RS // 2) + tl.arange(0, RS // 2)
    tl.store(fp4_ptr + 0 * stride_fp4_m + fp4_offs_n, packed)

    # Directly write sorted stage1 scale layout (decode M=1 path).
    num_valid = tl.load(num_valid_ids_ptr)
    q = tl.arange(0, MAX_Q)
    sid = tl.load(sorted_ids_ptr + q, mask=q < num_valid, other=token_num)
    tok = sid & 0xFFFFFF
    q_valid = (q < num_valid) & (tok < token_num)

    cols = col_base + tl.arange(0, NUM_QG)  # global scale columns this program owns
    col_valid = cols < n_i
    vals_row = tl.reshape(e8m0_u8, (NUM_QG,))
    vals = tl.broadcast_to(vals_row[None, :], (MAX_Q, NUM_QG))
    vals = tl.where(q_valid[:, None], vals, 0)

    pid_m_out = q // 32
    q_in = q % 32
    q_half = q_in // 16
    m_local = q_in % 16
    pid_n = cols // 8
    n_local = cols % 4
    half = (cols % 8) // 4
    i = q_half[:, None] + 2 * half[None, :]
    flat = (
        (((pid_m_out[:, None] * tile_n + pid_n[None, :]) * 4 + n_local[None, :]) * 16 + m_local[:, None]) * 4
        + i
    )
    dst_row = flat // n_i
    dst_col = flat % n_i

    mask = col_valid[None, :] & (q[:, None] < m_o)
    tl.store(
        sorted_scale_ptr + dst_row * stride_sc_m + dst_col * stride_sc_n,
        vals,
        mask=mask,
    )


@triton.jit
def _fused_decode_m1_topk8_k2048_kernel(
    x_ptr,
    rot_ptr,
    fp4_ptr,
    sorted_ids_ptr,
    num_valid_ids_ptr,
    sorted_scale_ptr,
    stride_x_m,
    stride_x_n,
    stride_rot_r,
    stride_rot_c,
    stride_fp4_m,
    stride_sc_m,
    stride_sc_n,
    token_num,
    m_o,
    MAX_Q: tl.constexpr,
):
    # Specialized decode kernel for common case:
    # K=2048, RS=128, QG=32, n_i=64, topk=8.
    RS: tl.constexpr = 128
    QG: tl.constexpr = 32
    NUM_QG: tl.constexpr = 4
    pid_k = tl.program_id(0)  # [0, 15]
    col_base = pid_k * NUM_QG

    offs_k = tl.arange(0, RS)
    x = tl.load(x_ptr + (pid_k * RS + offs_k) * stride_x_n).to(tl.float32)
    offs_r = tl.arange(0, RS)
    offs_c = tl.arange(0, RS)
    rot = tl.load(
        rot_ptr + offs_r[:, None] * stride_rot_r + offs_c[None, :] * stride_rot_c
    ).to(tl.float32)
    acc = tl.dot(x[None, :].to(tl.bfloat16), rot.to(tl.bfloat16))
    acc = acc.to(tl.bfloat16).to(tl.float32)

    HALF_QG: tl.constexpr = 16
    acc_g = tl.reshape(acc, (1, NUM_QG, QG))
    amax = tl.max(tl.abs(acc_g), axis=-1, keep_dims=False)

    amax_u32 = amax.to(tl.uint32, bitcast=True)
    amax_u32 = (amax_u32 + 0x200000) & 0xFF800000
    raw_exp = (amax_u32 >> 23) & 0xFF
    e8m0 = tl.maximum(raw_exp, 2) - 2
    e8m0_u8 = e8m0.to(tl.uint8)

    hw_scale = (e8m0.to(tl.uint32) << 23).to(tl.float32, bitcast=True)

    acc_pairs = tl.reshape(acc_g, (1, NUM_QG, HALF_QG, 2))
    even_vals, odd_vals = tl.split(acc_pairs)
    even_flat = tl.reshape(even_vals, (1, NUM_QG * HALF_QG))
    odd_flat = tl.reshape(odd_vals, (1, NUM_QG * HALF_QG))
    hw_scale_broad = tl.broadcast_to(
        tl.reshape(hw_scale, (1, NUM_QG, 1)), (1, NUM_QG, HALF_QG)
    )
    scale_flat = tl.reshape(hw_scale_broad, (1, NUM_QG * HALF_QG))

    packed_u32 = tl.inline_asm_elementwise(
        asm="v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3",
        constraints="=v,v,v,v",
        args=[even_flat, odd_flat, scale_flat],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    packed = packed_u32.to(tl.uint8)
    packed = tl.reshape(packed, (RS // 2,))

    fp4_offs_n = pid_k * (RS // 2) + tl.arange(0, RS // 2)
    tl.store(fp4_ptr + fp4_offs_n, packed)

    num_valid = tl.load(num_valid_ids_ptr)
    q = tl.arange(0, MAX_Q)
    sid = tl.load(sorted_ids_ptr + q, mask=q < num_valid, other=token_num)
    tok = sid & 0xFFFFFF
    q_valid = (q < num_valid) & (tok < token_num)

    cols = col_base + tl.arange(0, NUM_QG)
    vals_row = tl.reshape(e8m0_u8, (NUM_QG,))
    vals = tl.broadcast_to(vals_row[None, :], (MAX_Q, NUM_QG))
    vals = tl.where(q_valid[:, None], vals, 0)

    pid_m_out = q >> 5
    q_in = q & 31
    q_half = q_in >> 4
    m_local = q_in & 15
    pid_n = cols >> 3
    n_local = cols & 3
    half = (cols & 7) >> 2
    i = q_half[:, None] + (half[None, :] << 1)

    # For n_i=64: tile_n=8, dst_row=flat>>6, dst_col=flat&63.
    flat = (
        ((((pid_m_out[:, None] << 3) + pid_n[None, :]) << 2) + n_local[None, :]) * 16
        + m_local[:, None]
    ) * 4 + i
    dst_row = flat >> 6
    dst_col = flat & 63

    mask = q[:, None] < m_o
    tl.store(
        sorted_scale_ptr + dst_row * stride_sc_m + dst_col * stride_sc_n,
        vals,
        mask=mask,
    )


@triton.jit
def _fused_decode_m1_topk8_general_kernel(
    x_ptr,
    rot_ptr,
    fp4_ptr,
    sorted_ids_ptr,
    num_valid_ids_ptr,
    sorted_scale_ptr,
    stride_x_m,
    stride_x_n,
    stride_rot_r,
    stride_rot_c,
    stride_fp4_m,
    stride_sc_m,
    stride_sc_n,
    token_num,
    m_o,
    N_I: tl.constexpr,
    TILE_N: tl.constexpr,
    MAX_Q: tl.constexpr,
):
    """Generalized topk=8 decode kernel — n_i/tile_n as constexpr for fast div/mod."""
    RS: tl.constexpr = 128
    QG: tl.constexpr = 32
    NUM_QG: tl.constexpr = 4
    HALF_QG: tl.constexpr = 16
    pid_k = tl.program_id(0)
    col_base = pid_k * NUM_QG

    offs_k = tl.arange(0, RS)
    x = tl.load(x_ptr + (pid_k * RS + offs_k) * stride_x_n).to(tl.float32)
    offs_r = tl.arange(0, RS)
    offs_c = tl.arange(0, RS)
    rot = tl.load(
        rot_ptr + offs_r[:, None] * stride_rot_r + offs_c[None, :] * stride_rot_c
    ).to(tl.float32)
    acc = tl.dot(x[None, :].to(tl.bfloat16), rot.to(tl.bfloat16))
    acc = acc.to(tl.bfloat16).to(tl.float32)

    acc_g = tl.reshape(acc, (1, NUM_QG, QG))
    amax = tl.max(tl.abs(acc_g), axis=-1, keep_dims=False)

    amax_u32 = amax.to(tl.uint32, bitcast=True)
    amax_u32 = (amax_u32 + 0x200000) & 0xFF800000
    raw_exp = (amax_u32 >> 23) & 0xFF
    e8m0 = tl.maximum(raw_exp, 2) - 2
    e8m0_u8 = e8m0.to(tl.uint8)

    hw_scale = (e8m0.to(tl.uint32) << 23).to(tl.float32, bitcast=True)

    acc_pairs = tl.reshape(acc_g, (1, NUM_QG, HALF_QG, 2))
    even_vals, odd_vals = tl.split(acc_pairs)
    even_flat = tl.reshape(even_vals, (1, NUM_QG * HALF_QG))
    odd_flat = tl.reshape(odd_vals, (1, NUM_QG * HALF_QG))
    hw_scale_broad = tl.broadcast_to(
        tl.reshape(hw_scale, (1, NUM_QG, 1)), (1, NUM_QG, HALF_QG)
    )
    scale_flat = tl.reshape(hw_scale_broad, (1, NUM_QG * HALF_QG))

    packed_u32 = tl.inline_asm_elementwise(
        asm="v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3",
        constraints="=v,v,v,v",
        args=[even_flat, odd_flat, scale_flat],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    packed = packed_u32.to(tl.uint8)
    packed = tl.reshape(packed, (RS // 2,))

    fp4_offs_n = pid_k * (RS // 2) + tl.arange(0, RS // 2)
    tl.store(fp4_ptr + fp4_offs_n, packed)

    num_valid = tl.load(num_valid_ids_ptr)
    q = tl.arange(0, MAX_Q)
    sid = tl.load(sorted_ids_ptr + q, mask=q < num_valid, other=token_num)
    tok = sid & 0xFFFFFF
    q_valid = (q < num_valid) & (tok < token_num)

    cols = col_base + tl.arange(0, NUM_QG)
    col_valid = cols < N_I
    vals_row = tl.reshape(e8m0_u8, (NUM_QG,))
    vals = tl.broadcast_to(vals_row[None, :], (MAX_Q, NUM_QG))
    vals = tl.where(q_valid[:, None], vals, 0)

    pid_m_out = q >> 5
    q_in = q & 31
    q_half = q_in >> 4
    m_local = q_in & 15
    pid_n = cols >> 3
    n_local = cols & 3
    half = (cols & 7) >> 2
    i = q_half[:, None] + (half[None, :] << 1)

    flat = (
        (((pid_m_out[:, None] * TILE_N + pid_n[None, :]) * 4 + n_local[None, :]) * 16
         + m_local[:, None]) * 4 + i
    )
    dst_row = flat // N_I
    dst_col = flat % N_I

    mask = col_valid[None, :] & (q[:, None] < m_o)
    tl.store(
        sorted_scale_ptr + dst_row * stride_sc_m + dst_col * stride_sc_n,
        vals,
        mask=mask,
    )


@triton.jit
def _stage1_scale_from_shuffled_kernel(
    shuffled_ptr,
    sorted_ids_ptr,
    num_valid_ids_ptr,
    out_ptr,
    stride_sh_m,
    stride_sh_n,
    stride_out_m,
    stride_out_n,
    token_num,
    n_i,
    sn_padded,
    tile_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_valid = tl.load(num_valid_ids_ptr)
    if pid_m * BLOCK_M >= num_valid:
        return

    m_local = tl.arange(0, 16)[:, None]
    n_local = tl.arange(0, 4)[None, :]

    for i in tl.static_range(0, 4):
        q = pid_m * BLOCK_M + (i % 2) * 16 + m_local
        src_n = pid_n * BLOCK_N + (i // 2) * 4 + n_local
        sid = tl.load(sorted_ids_ptr + q, mask=q < num_valid, other=token_num)
        tok = sid & 0xFFFFFF

        tok_valid = tok < token_num
        src_valid = src_n < n_i
        load_mask = tok_valid & src_valid

        tok_local = tok % 32
        tok_base = (tok // 32) * 32
        flat_idx = (
            (src_n // 8) * 256
            + (src_n % 4) * 64
            + (tok_local % 16) * 4
            + ((src_n % 8) // 4) * 2
            + (tok_local // 16)
        )
        sh_row = (flat_idx // sn_padded) + tok_base
        sh_col = flat_idx % sn_padded

        v = tl.load(
            shuffled_ptr + sh_row * stride_sh_m + sh_col * stride_sh_n,
            mask=load_mask,
            other=0,
        )
        flat = (((pid_m * tile_n + pid_n) * 4 + n_local) * 16 + m_local) * 4 + i
        dst_row = flat // n_i
        dst_col = flat % n_i
        tl.store(
            out_ptr + dst_row * stride_out_m + dst_col * stride_out_n,
            v,
            mask=src_valid,
        )


def _build_stage1_scale_from_shuffled(
    shuffled_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    n_i: int,
) -> torch.Tensor:
    m_o = sorted_ids.shape[0]
    m_pad = ((m_o + 31) // 32) * 32
    out_u8 = torch.empty((m_pad, n_i), dtype=torch.uint8, device=shuffled_scale.device)
    grid = (triton.cdiv(m_o, 32), triton.cdiv(n_i, 8))
    _stage1_scale_from_shuffled_kernel[grid](
        shuffled_scale,
        sorted_ids,
        num_valid_ids,
        out_u8,
        shuffled_scale.stride(0),
        shuffled_scale.stride(1),
        out_u8.stride(0),
        out_u8.stride(1),
        token_num=token_num,
        n_i=n_i,
        sn_padded=shuffled_scale.shape[1],
        tile_n=triton.cdiv(n_i, 8),
        BLOCK_M=32,
        BLOCK_N=8,
        num_warps=1,
    )
    return out_u8.view(dtypes.fp8_e8m0)


def _fused_rot_quant_moe_sort_impl(
    x: torch.Tensor,
    rotation: torch.Tensor,
    rotation_size: int,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    topk: int,
    block_size: int = 32,
    use_hip_kernel: bool | None = None,
    use_triton_rot_sort_kernel: bool | None = None,
):
    """Core implementation — called by both the custom op and the public API."""
    M, K = x.shape
    RS = rotation_size
    QG = 32
    if use_hip_kernel is None:
        use_hip_kernel = ENABLE_HIP_FUSED_ROTATION
    if use_triton_rot_sort_kernel is None:
        use_triton_rot_sort_kernel = ENABLE_TRITON_ROT_SORT_FUSION

    # HIP MFMA 3-in-1: single kernel for all M, ~12-15µs
    if ENABLE_HIP_MFMA and RS == 128 and (K % RS == 0):
        from vllm.model_executor.layers.quantization.quark.fused_rotation_quant_mfma_hip import (
            fused_mfma_rot_quant_moe_sort, is_available,
        )
        if is_available():
            key = ("hip_mfma", M, K, RS, token_num, topk, block_size)
            if key not in _DISPATCH_LOG_KEYS:
                _DISPATCH_LOG_KEYS.add(key)
                logger.info("fused_rot_quant_moe dispatch: HIP MFMA 3-in-1 M=%s K=%s", M, K)
            return fused_mfma_rot_quant_moe_sort(
                x, rotation, RS,
                sorted_ids, num_valid_ids,
                token_num, topk, block_size,
            )

    # Keep this experimental path opt-in until it beats gluon+moe_mxfp4_sort
    # in end-to-end measurements.
    use_gluon_sorted_fusion = (
        ENABLE_GLUON_SORTED_SCALE_FUSION
        and topk in (1, 8)
        and token_num == 1
        and M == 1
    )
    # HIP path corresponds to the "fused" 2-kernel pipeline (rot+quant, then sort),
    # so it should not enter gluon single-kernel sorted-scale fusion branch.
    if use_hip_kernel:
        use_gluon_sorted_fusion = False
    use_gluon_moe_decode = (
        ENABLE_GLUON_MOE_DECODE
        and (not use_hip_kernel)
        and (not use_gluon_sorted_fusion)
        and M == 1
        and token_num == 1
        and topk in (1, 8)
        and RS == 128
        and (K % RS == 0)
    )
    use_triton_decode_rot_sort = (
        use_triton_rot_sort_kernel
        and (not use_hip_kernel)
        and (not use_gluon_sorted_fusion)
        and (not use_gluon_moe_decode)
        and M == 1
        and token_num == 1
        and topk in (1, 8)
    )
    use_gluon_kw8 = (
        ENABLE_GLUON_KW8
        and (not use_hip_kernel)
        and (not use_gluon_sorted_fusion)
        and (not use_gluon_moe_decode)
        and (not use_triton_decode_rot_sort)
        and RS == 128
        and (K % RS == 0)
    )
    key = (M, K, RS, token_num, topk, use_hip_kernel,
           use_gluon_moe_decode, use_triton_decode_rot_sort, use_gluon_kw8, block_size)
    if key not in _DISPATCH_LOG_KEYS:
        _DISPATCH_LOG_KEYS.add(key)
        logger.debug(
            "fused_rot_quant_moe dispatch: M=%s K=%s gluon_moe=%s triton_m1=%s gluon_kw8=%s",
            M, K, use_gluon_moe_decode, use_triton_decode_rot_sort, use_gluon_kw8,
        )
    if use_gluon_moe_decode:
        from vllm.model_executor.layers.quantization.quark.fused_rotation_mxfp4_quant_moe_gluon import (
            fused_decode_m1_rot_quant_sorted_gluon,
        )
        n_i = K // QG
        m_o = sorted_ids.shape[0]
        m_pad = ((m_o + 31) // 32) * 32
        fp4_u8 = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
        sorted_u8 = torch.empty((m_pad + 1, n_i), dtype=torch.uint8, device=x.device)
        fused_decode_m1_rot_quant_sorted_gluon(
            x, rotation, fp4_u8,
            sorted_ids, num_valid_ids, sorted_u8,
            K=K, RS=RS, QG=QG, n_i=n_i,
            token_num=token_num, m_o=m_o, m_pad=m_pad,
        )
        return fp4_u8.view(dtypes.fp4x2), sorted_u8[:m_pad].view(dtypes.fp8_e8m0)

    if use_triton_decode_rot_sort:
        n_i = K // QG
        m_o = sorted_ids.shape[0]
        m_pad = ((m_o + 31) // 32) * 32
        max_q = _pick_decode_max_q(m_pad)
        fp4_u8 = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
        sorted_u8 = torch.empty((m_pad, n_i), dtype=torch.uint8, device=x.device)
        grid = (K // RS,)
        use_topk8_k2048 = (
            topk == 8
            and K == 2048
            and RS == 128
            and n_i == 64
            and max_q <= 256
        )
        use_topk8_general = (
            topk == 8
            and RS == 128
            and max_q <= 256
            and not use_topk8_k2048
        )
        if use_topk8_k2048:
            _fused_decode_m1_topk8_k2048_kernel[grid](
                x,
                rotation,
                fp4_u8,
                sorted_ids,
                num_valid_ids,
                sorted_u8,
                x.stride(0),
                x.stride(1),
                rotation.stride(0),
                rotation.stride(1),
                fp4_u8.stride(0),
                sorted_u8.stride(0),
                sorted_u8.stride(1),
                token_num=token_num,
                m_o=m_o,
                MAX_Q=max_q,
                num_warps=1,
            )
        elif use_topk8_general:
            _fused_decode_m1_topk8_general_kernel[grid](
                x,
                rotation,
                fp4_u8,
                sorted_ids,
                num_valid_ids,
                sorted_u8,
                x.stride(0),
                x.stride(1),
                rotation.stride(0),
                rotation.stride(1),
                fp4_u8.stride(0),
                sorted_u8.stride(0),
                sorted_u8.stride(1),
                token_num=token_num,
                m_o=m_o,
                N_I=n_i,
                TILE_N=triton.cdiv(n_i, 8),
                MAX_Q=max_q,
                num_warps=1,
            )
        else:
            _fused_decode_m1_rot_quant_sorted_kernel[grid](
                x,
                rotation,
                fp4_u8,
                sorted_ids,
                num_valid_ids,
                sorted_u8,
                x.stride(0),
                x.stride(1),
                rotation.stride(0),
                rotation.stride(1),
                fp4_u8.stride(0),
                sorted_u8.stride(0),
                sorted_u8.stride(1),
                token_num=token_num,
                n_i=n_i,
                tile_n=triton.cdiv(n_i, 8),
                m_o=m_o,
                RS=RS,
                QG=QG,
                MAX_Q=max_q,
                num_warps=4,
            )
        return fp4_u8.view(dtypes.fp4x2), sorted_u8.view(dtypes.fp8_e8m0)
    if use_gluon_sorted_fusion:
        n_scales = K // QG
        sn_padded = ((n_scales + 7) // 8) * 8
        m_pad = ((sorted_ids.shape[0] + 31) // 32) * 32
        fp4_u8 = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
        sorted_u8 = torch.empty((m_pad, sn_padded), dtype=torch.uint8, device=x.device)
        fp4_u8, sorted_u8 = fused_gluon_v2(
            x,
            rotation,
            RS,
            fp4_out=fp4_u8,
            scales_out=sorted_u8,
            sorted_scales_topk8=True,
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=token_num,
        )
        sorted_scale = sorted_u8[:, :n_scales].view(dtypes.fp8_e8m0)
    else:
        fp4_u8 = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
        raw_scale = torch.empty((M, K // QG), dtype=torch.uint8, device=x.device)
        if use_hip_kernel:
            from vllm.model_executor.layers.quantization.quark.fused_rotation_quant_hip import (
                fused_rotation_quant_hip,
            )
            fp4_u8, raw_scale = fused_rotation_quant_hip(
                x, rotation, RS,
                fp4_out=fp4_u8, scales_out=raw_scale, shuffle_scales=False,
            )
        elif use_gluon_kw8:
            fp4_u8, raw_scale = fused_gluon_v2_kw8(
                x, rotation, RS, fp4_out=fp4_u8, scales_out=raw_scale, shuffle_scales=False
            )
        else:
            fp4_u8, raw_scale = fused_gluon_v2(
                x, rotation, RS, fp4_out=fp4_u8, scales_out=raw_scale, shuffle_scales=False
            )
        sorted_scale = moe_mxfp4_sort(
            raw_scale, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
            token_num=token_num, block_size=block_size,
        )

    return fp4_u8.view(dtypes.fp4x2), sorted_scale


def fused_rotation_mxfp4_quant_moe_sort(
    x: torch.Tensor,
    rotation: torch.Tensor,
    rotation_size: int,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    topk: int,
    block_size: int = 32,
    use_hip_kernel: bool | None = None,
    use_triton_rot_sort_kernel: bool | None = None,
):
    """
    Fused rotation + MXFP4 quant + MoE sort.
    Uses custom op when available (CUDAGraph compatible).
    """
    if (
        _has_moe_rot_quant_custom_op
        and use_hip_kernel is None
        and use_triton_rot_sort_kernel is None
    ):
        return torch.ops.vllm.fused_rotation_mxfp4_quant_moe_sort(
            x, rotation, rotation_size,
            sorted_ids, num_valid_ids,
            token_num, topk, block_size,
        )
    return _fused_rot_quant_moe_sort_impl(
        x, rotation, rotation_size,
        sorted_ids, num_valid_ids,
        token_num, topk, block_size,
        use_hip_kernel=use_hip_kernel,
        use_triton_rot_sort_kernel=use_triton_rot_sort_kernel,
    )
