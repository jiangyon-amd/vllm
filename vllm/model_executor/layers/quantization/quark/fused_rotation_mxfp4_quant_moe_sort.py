"""
Fused Rotation + MXFP4 Quant + MoE Sort.

Replaces: torch.matmul(rotation) + fused_dynamic_mxfp4_quant_moe_sort  [2 kernels + bf16 GMEM]
With:     Gluon rotation+quant + moe_mxfp4_sort                        [2 kernels, no bf16 GMEM]

Same kernel count, saves bf16 intermediate GMEM read/write.
"""

import os
import logging
import json
import time
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
ENABLE_TRITON_ROT_SORT_FUSION = (
    os.getenv("VLLM_MOE_TRITON_ROT_SORT_FUSION", "0") == "1"
)
ENABLE_GLUON_KW8 = (
    os.getenv("VLLM_MOE_GLUON_KW8", "1") == "1"
)
ENABLE_FUSED_MOE_DEBUG_LOG = (
    os.getenv("VLLM_MOE_FUSED_DEBUG_LOG", "0") == "1"
)

logger = logging.getLogger(__name__)
_DISPATCH_LOG_KEYS: set[tuple] = set()
_TRITON_PROTO_LOG_KEYS: set[tuple] = set()
_DEBUG_LOG_PATH = "/data/jiangyon/.cursor/debug-d7954d.log"


def _agent_debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": "d7954d",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _pick_decode_max_q(m_pad: int) -> int:
    # Fixed at 256: kernel processes first MAX_Q sorted_ids entries.
    # num_valid_ids is typically << 256 for decode (M=1).
    # Keeping MAX_Q small avoids bloating the scatter loop.
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

    # MXFP4 quantization.
    acc_g = tl.reshape(acc, (1, NUM_QG, QG))
    amax = tl.max(tl.abs(acc_g), axis=-1, keep_dims=False)
    # Match aiter separated kernel's scale derivation exactly.
    amax_i32 = amax.to(tl.int32, bitcast=True)
    amax_u32 = (amax_i32 + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax_fp = amax_u32.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax_fp).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
    e8m0_u8 = scale_e8m0_unbiased.to(tl.uint8) + 127  # [1, NUM_QG]
    quant_scale = tl.exp2(-scale_e8m0_unbiased)
    quant_scale_full = tl.reshape(
        tl.broadcast_to(tl.reshape(quant_scale, (1, NUM_QG, 1)), (1, NUM_QG, QG)),
        (1, RS),
    )
    qx = (acc * quant_scale_full).to(tl.uint32, bitcast=True)
    s = qx & 0x80000000
    e = (qx >> 23) & 0xFF
    m = qx & 0x7FFFFF
    E8_BIAS: tl.constexpr = 127
    E2_BIAS: tl.constexpr = 1
    adjusted_exponents = tl.core.sub(E8_BIAS, e + 1, sanitize_overflow=False)
    m = tl.where(e < E8_BIAS, (0x400000 | (m >> 1)) >> adjusted_exponents, m)
    e = tl.maximum(e, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)
    e2m1_tmp = tl.minimum((((e << 2) | (m >> 21)) + 1) >> 1, 0x7)
    e2m1 = ((s >> 28) | e2m1_tmp).to(tl.uint8)
    e2m1 = tl.reshape(e2m1, (1, RS // 2, 2))
    even, odd = tl.split(e2m1)
    packed = even | (odd << 4)
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

    acc_g = tl.reshape(acc, (1, NUM_QG, QG))
    amax = tl.max(tl.abs(acc_g), axis=-1, keep_dims=False)
    amax_i32 = amax.to(tl.int32, bitcast=True)
    amax_u32 = (amax_i32 + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax_fp = amax_u32.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax_fp).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
    e8m0_u8 = scale_e8m0_unbiased.to(tl.uint8) + 127

    quant_scale = tl.exp2(-scale_e8m0_unbiased)
    quant_scale_full = tl.reshape(
        tl.broadcast_to(tl.reshape(quant_scale, (1, NUM_QG, 1)), (1, NUM_QG, QG)),
        (1, RS),
    )
    qx = (acc * quant_scale_full).to(tl.uint32, bitcast=True)
    s = qx & 0x80000000
    e = (qx >> 23) & 0xFF
    m = qx & 0x7FFFFF
    E8_BIAS: tl.constexpr = 127
    E2_BIAS: tl.constexpr = 1
    adjusted_exponents = tl.core.sub(E8_BIAS, e + 1, sanitize_overflow=False)
    m = tl.where(e < E8_BIAS, (0x400000 | (m >> 1)) >> adjusted_exponents, m)
    e = tl.maximum(e, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)
    e2m1_tmp = tl.minimum((((e << 2) | (m >> 21)) + 1) >> 1, 0x7)
    e2m1 = ((s >> 28) | e2m1_tmp).to(tl.uint8)
    e2m1 = tl.reshape(e2m1, (1, RS // 2, 2))
    even, odd = tl.split(e2m1)
    packed = tl.reshape(even | (odd << 4), (RS // 2,))

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
    use_triton_decode_rot_sort = (
        use_triton_rot_sort_kernel
        and (not use_hip_kernel)
        and (not use_gluon_sorted_fusion)
        and M == 1
        and token_num == 1
        and topk in (1, 8)
    )
    use_gluon_kw8 = (
        ENABLE_GLUON_KW8
        and (not use_hip_kernel)
        and (not use_gluon_sorted_fusion)
        and (not use_triton_decode_rot_sort)
        and RS == 128
        and (K % RS == 0)
    )
    use_topk8_decode = topk == 8
    if ENABLE_FUSED_MOE_DEBUG_LOG:
        key = (
            M,
            K,
            RS,
            token_num,
            topk,
            use_hip_kernel,
            use_gluon_sorted_fusion,
            use_triton_decode_rot_sort,
            use_gluon_kw8,
            use_topk8_decode,
            block_size,
        )
        if key not in _DISPATCH_LOG_KEYS:
            _DISPATCH_LOG_KEYS.add(key)
            logger.info(
                "fused_rotation_mxfp4_quant_moe_sort dispatch: M=%s K=%s RS=%s token_num=%s topk=%s "
                "hip=%s gluon_sorted=%s triton_decode=%s gluon_kw8=%s topk8=%s block_size=%s valid_ids=%s",
                M,
                K,
                RS,
                token_num,
                topk,
                use_hip_kernel,
                use_gluon_sorted_fusion,
                use_triton_decode_rot_sort,
                use_gluon_kw8,
                use_topk8_decode,
                block_size,
                int(num_valid_ids[0].item()) if num_valid_ids.numel() > 0 else -1,
            )
            # #region agent log
            _agent_debug_log(
                run_id="separated-path-debug",
                hypothesis_id="H2",
                location="fused_rotation_mxfp4_quant_moe_sort.py:fused_rotation_mxfp4_quant_moe_sort",
                message="Fused wrapper dispatch",
                data={
                    "M": int(M),
                    "K": int(K),
                    "RS": int(RS),
                    "token_num": int(token_num),
                    "topk": int(topk),
                    "use_hip_kernel": bool(use_hip_kernel),
                    "use_gluon_sorted_fusion": bool(use_gluon_sorted_fusion),
                    "use_triton_decode_rot_sort": bool(use_triton_decode_rot_sort),
                    "use_gluon_kw8": bool(use_gluon_kw8),
                    "use_topk8_decode": bool(use_topk8_decode),
                    "block_size": int(block_size),
                    "valid_ids": int(num_valid_ids[0].item()) if num_valid_ids.numel() > 0 else -1,
                },
            )
            # #endregion
    if use_triton_decode_rot_sort:
        n_i = K // QG
        m_o = sorted_ids.shape[0]
        m_pad = ((m_o + 31) // 32) * 32
        max_q = _pick_decode_max_q(m_pad)
        fp4_u8 = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
        sorted_u8 = torch.empty((m_pad, n_i), dtype=torch.uint8, device=x.device)
        grid = (K // RS,)
        use_topk8_special = (
            topk == 8
            and K == 2048
            and RS == 128
            and n_i == 64
            and max_q <= 256
        )
        if ENABLE_FUSED_MOE_DEBUG_LOG:
            proto_key = (M, K, RS, token_num, topk, m_o, n_i, max_q)
            if proto_key not in _TRITON_PROTO_LOG_KEYS:
                _TRITON_PROTO_LOG_KEYS.add(proto_key)
                # #region agent log
                _agent_debug_log(
                    run_id="single-kernel-prototype",
                    hypothesis_id="H3",
                    location="fused_rotation_mxfp4_quant_moe_sort.py:fused_rotation_mxfp4_quant_moe_sort",
                    message="Select Triton decode single-kernel rot+quant+sort path",
                    data={
                        "M": int(M),
                        "K": int(K),
                        "RS": int(RS),
                        "token_num": int(token_num),
                        "topk": int(topk),
                        "m_o": int(m_o),
                        "m_pad": int(m_pad),
                        "max_q": int(max_q),
                        "n_i": int(n_i),
                        "grid_k": int(grid[0]),
                        "use_topk8_special": bool(use_topk8_special),
                    },
                )
                # #endregion
        if use_topk8_special:
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
                num_warps=4,
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
                x,
                rotation,
                RS,
                fp4_out=fp4_u8,
                scales_out=raw_scale,
                shuffle_scales=False,
            )
        else:
            if use_gluon_kw8:
                fp4_u8, raw_scale = fused_gluon_v2_kw8(
                    x, rotation, RS, fp4_out=fp4_u8, scales_out=raw_scale, shuffle_scales=False
                )
            else:
                fp4_u8, raw_scale = fused_gluon_v2(
                    x, rotation, RS, fp4_out=fp4_u8, scales_out=raw_scale, shuffle_scales=False
                )
        sorted_scale = moe_mxfp4_sort(
            raw_scale,
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=token_num,
            block_size=block_size,
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
