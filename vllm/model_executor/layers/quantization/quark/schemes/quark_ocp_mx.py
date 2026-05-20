# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from fractions import Fraction
import os
from functools import cache, partial
from typing import Any

import torch
import torch.nn.functional as F

from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.quark.transform import (
    OrthogonalTransform,
    build_hadamard_matrix,
    rotation_weight_loader,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    dequant_mxfp4,
    quant_dequant_mxfp4,
)
from vllm.model_executor.layers.quantization.utils.mxfp6_utils import (
    dequant_mxfp6,
    quant_dequant_mxfp6,
)
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    OCP_MX_BLOCK_SIZE,
    OCP_MX_Scheme,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PackedvLLMParameter,
)
from vllm.platforms import current_platform

from .quark_scheme import QuarkScheme

# Fused rotation+quant layout (align with OCP_MX_BLOCK_SIZE=32)
FP4_ELEMS_PER_BYTE = 2
SCALE_COL_ALIGN = 8
SCALE_ROW_TILE = 256
DECODE_MAX_M = OCP_MX_BLOCK_SIZE  # decode path: M < DECODE_MAX_M uses raw scales

# ---------------------------------------------------------------------------
# Fused rotation + MXFP4 quantization — registered as custom op.
#
# HIP MFMA dense kernel is 2-3x faster than Gluon Triton at kernel level.
# Requires aiter >= 0.1 with correct mutates_args registration so that
# torch.compile/CUDAGraph sees x and rotation as read-only.
# Use VLLM_DENSE_HIP_MFMA=0 to force Gluon Triton fallback.
# ---------------------------------------------------------------------------
_has_fused_triton_rot_quant = False
_use_hip_mfma_dense = False

# HIP MFMA dense kernel — enabled by default (2-3x faster, CUDAGraph safe
# after aiter mutates_args fix)
if os.environ.get("VLLM_DENSE_HIP_MFMA", "1") == "1":
    try:
        from aiter.ops.mfma_rot_quant_dense import (
            mfma_rot_quant_dense as _hip_mfma_rot_quant_dense,
        )
        _use_hip_mfma_dense = True
    except ImportError:
        _hip_mfma_rot_quant_dense = None

# Gluon Triton fallback
try:
    from vllm.model_executor.layers.quantization.quark.fused_rotation_quant_gluon import (
        fused_rot_quant_gluon as _fused_rot_quant_gluon,
    )
except ImportError:
    _fused_rot_quant_gluon = None

if _use_hip_mfma_dense or _fused_rot_quant_gluon is not None:
    from vllm.utils.torch_utils import direct_register_custom_op

    # Pre-allocated static buffers to eliminate per-call GPU allocations.
    # Key: (M, K, device_index) → (fp4_buf, sc_buf)
    # CUDAGraph-safe: same buffer address across captures and replays.
    _rot_quant_buf_cache: dict = {}

    def _get_rot_quant_bufs(
        M: int, K: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pre-allocated (fp4, sc) buffers, creating them on first call."""
        key = (M, K, device.index if device.index is not None else 0)
        if key not in _rot_quant_buf_cache:
            n_sc = K // 32
            sn_padded = (n_sc + 7) // 8 * 8
            sm_padded = (M + 255) // 256 * 256
            fp4 = torch.empty((M, K // 2), dtype=torch.uint8, device=device)
            # sc is always zero-initialized once; kernel writes all live cells.
            sc = torch.zeros((sm_padded, sn_padded), dtype=torch.uint8, device=device)
            _rot_quant_buf_cache[key] = (fp4, sc)
        return _rot_quant_buf_cache[key]

    def _fused_rot_quant_op(
        x: torch.Tensor,
        rotation: torch.Tensor,
        rotation_size: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_2d = x.reshape(-1, x.shape[-1])
        M, K = x_2d.shape

        if _use_hip_mfma_dense:
            shuffle = M >= 32
            fp4, sc = _get_rot_quant_bufs(M, K, x.device)
            # sc padding cells stay zero from one-time initialization.
            # No sc.zero_() needed on every call — the kernel only writes
            # live cells and the GEMM downstream reads only those cells.
            _hip_mfma_rot_quant_dense(
                x_2d, rotation, fp4, sc, rotation_size, shuffle)
        elif _fused_rot_quant_gluon is not None:
            n_sc = K // 32
            sn_padded = (n_sc + 7) // 8 * 8
            sm_padded = (M + 255) // 256 * 256
            fp4 = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
            sc = torch.empty((sm_padded, sn_padded), dtype=torch.uint8, device=x.device)
            if M < 32:
                sc_raw = torch.empty(
                    (M, n_sc), dtype=torch.uint8, device=x.device)
                _fused_rot_quant_gluon(
                    x_2d, rotation, rotation_size,
                    fp4_out=fp4, scales_out=sc_raw, shuffle_scales=False)
                sc[:M, :n_sc] = sc_raw
            else:
                sc.zero_()
                _fused_rot_quant_gluon(
                    x_2d, rotation, rotation_size,
                    fp4_out=fp4, scales_out=sc, shuffle_scales=True)
        return fp4, sc

    def _fused_rot_quant_fake(
        x: torch.Tensor,
        rotation: torch.Tensor,
        rotation_size: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        M = x.reshape(-1, x.shape[-1]).shape[0]
        K = x.shape[-1]
        sn_padded = (K // 32 + 7) // 8 * 8
        sm_padded = (M + 255) // 256 * 256
        return (
            torch.empty((M, K // 2), dtype=torch.uint8, device=x.device),
            torch.empty((sm_padded, sn_padded), dtype=torch.uint8, device=x.device),
        )

    direct_register_custom_op(
        op_name="fused_rotation_mxfp4_quant",
        op_func=_fused_rot_quant_op,
        mutates_args=[],
        fake_impl=_fused_rot_quant_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    def _fused_rot_quant(x, rotation, rotation_size=128, **kwargs):
        return torch.ops.vllm.fused_rotation_mxfp4_quant(x, rotation, rotation_size)

    _has_fused_triton_rot_quant = True
    _have_hip = _use_hip_mfma_dense
    _have_glu = _fused_rot_quant_gluon is not None
    # Dispatch policy: Gluon Triton is optimal for ALL M values based on
    # real E2E shape benchmarks on MI355X (K=2048/7168, M=1..1024).
    # HIP MFMA fallback when Gluon unavailable.
    # Use VLLM_HIP_FUSED_MAX_M=N to force HIP for M<N (default 0 = Gluon always).
    _hip_max = int(os.environ.get("VLLM_HIP_FUSED_MAX_M", "0"))
    if _have_glu and _hip_max == 0:
        _backend = "Gluon Triton (optimal for all M, 20-105% faster than Separated)"
    elif _have_hip and _have_glu:
        _backend = f"Hybrid HIP MFMA (M<{_hip_max}) + Gluon Triton (M≥{_hip_max})"
    elif _have_hip:
        _backend = "HIP MFMA only (Gluon unavailable)"
    else:
        _backend = "Gluon Triton only"
    print(f"[INFO] Registered fused_rotation_mxfp4_quant custom op ({_backend})")

logger = init_logger(__name__)



# TODO: move registration of custom op to aiter_ops.py
# `from vllm._aiter_ops import rocm_aiter_ops`
# use `rocm_aiter_ops.is_asm_fp4_gemm_dynamic_quant_enabled()`
# for envs checks which does not require @cache anymore.
# triton kernel is torch compile compatible.
# does not require direct registration.
# use `rocm_aiter_ops.triton_fp4_gemm_dynamic_qaunt`.
@cache
def is_rocm_aiter_fp4_asm_gemm_enabled() -> bool:
    return (
        current_platform.is_rocm()
        and envs.VLLM_ROCM_USE_AITER_FP4_ASM_GEMM
        and envs.VLLM_ROCM_USE_AITER
    )


def is_fused_rotation_quant_enabled() -> bool:
    """Switch for Dense rotation+quant: fused (one kernel) vs separated (matmul then quant).
    Default 0 to avoid fused when there is no rotation (e.g. RTN); set 1 for fused on
    rotation models (Hadamard or trained both have rotation matrices).
    - VLLM_DISABLE_FUSED_ROT_QUANT=1 or true → separated (backward compat)
    - VLLM_USE_FUSED_ROTATION_QUANT=1 or true → fused; =0 or false → separated
    - Default: 0 (separated). Use 1 for Hadamard/trained when fused is desired."""
    if os.environ.get("VLLM_DISABLE_FUSED_ROT_QUANT", "").strip().lower() in ("1", "true"):
        return False
    v = os.environ.get("VLLM_USE_FUSED_ROTATION_QUANT", "0").strip().lower()
    return v in ("1", "true")


try:
    from aiter.ops.shuffle import shuffle_weight
    from aiter.ops.triton.gemm_afp4wfp4 import (
        gemm_afp4wfp4,
        gemm_afp4wfp4_preshuffled_weight_scales,
    )
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    from vllm.utils.torch_utils import direct_register_custom_op

    if is_rocm_aiter_fp4_asm_gemm_enabled():
        from aiter import gemm_a4w4, per_1x32_f4_quant_hip

    # ------------------------------------------------------------------
    # Buffer cache for gemm_with_dynamic_quant fused rotation+quant path.
    # Pre-allocated per (M, K, layout, device) to eliminate per-call
    # torch.empty + sc_buf.zero_() overhead (~6µs/call).
    # CUDAGraph-safe: same buffer address across captures and replays.
    # ------------------------------------------------------------------
    _gemm_rot_quant_buf_cache: dict = {}

    def _get_gemm_rot_quant_bufs(
        M: int, K_in: int, is_tuned_decode: bool, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached (fp4_buf, sc_buf) for the given (M, K, layout)."""
        key = (M, K_in, bool(is_tuned_decode),
               device.index if device.index is not None else 0)
        cached = _gemm_rot_quant_buf_cache.get(key)
        if cached is not None:
            return cached
        n_scale_groups = K_in // OCP_MX_BLOCK_SIZE
        fp4_cols = K_in // FP4_ELEMS_PER_BYTE
        fp4_buf = torch.empty((M, fp4_cols), dtype=torch.uint8, device=device)
        if is_tuned_decode:
            # Tuned decode: no padding, shuffle disabled, no zero_() needed
            sc_buf = torch.empty(
                (M, n_scale_groups), dtype=torch.uint8, device=device)
        else:
            # Padded layout: HIP MFMA needs zero padding cells; init once
            sn_pad = (n_scale_groups + SCALE_COL_ALIGN - 1) // SCALE_COL_ALIGN * SCALE_COL_ALIGN
            sm_pad = (M + SCALE_ROW_TILE - 1) // SCALE_ROW_TILE * SCALE_ROW_TILE
            sc_buf = torch.zeros(
                (sm_pad, sn_pad), dtype=torch.uint8, device=device)
        _gemm_rot_quant_buf_cache[key] = (fp4_buf, sc_buf)
        return fp4_buf, sc_buf

    # ------------------------------------------------------------------
    # Dispatch policy for dense rotation+quant: HIP MFMA vs Gluon Triton.
    #
    # Real E2E shape benchmark (MI355X, CUDAGraph mode, Qwen3-30B-A3B):
    #   K=2048 (attention):
    #     M=1  HIP 4.60µs / Gluon 4.79µs → within 5% (Gluon preferred)
    #     M≥2  Gluon 4.85-7.3µs vs HIP 5.2-8.2µs → Gluon 7-42% faster
    #   K=7168 (MoE expert):
    #     M=1  HIP 4.73µs / Gluon 4.88µs → within 5% (Gluon preferred)
    #     M≥2  Gluon 4.93-16.3µs vs HIP 5.3-18.3µs → Gluon 7-31% faster
    #
    # Conclusion: Gluon Triton is optimal for ALL M values (HIP never
    # clearly dominates by >5% margin across real production shapes).
    # Speedup vs Separated (matmul+quant): Gluon 20-105% faster.
    #
    # Strategy: always use Gluon Triton when available.
    # Default HIP_FUSED_MAX_M=0 means Gluon for all M.
    # Override via env: VLLM_HIP_FUSED_MAX_M (set to 9999 to always use HIP,
    #                                          any value N → HIP if M < N).
    # ------------------------------------------------------------------
    HIP_FUSED_MAX_M = int(os.environ.get("VLLM_HIP_FUSED_MAX_M", "0"))
    _has_hip_mfma  = _use_hip_mfma_dense and (_hip_mfma_rot_quant_dense is not None)
    _has_gluon     = _fused_rot_quant_gluon is not None

    def gemm_with_dynamic_quant(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        rocm_use_aiter_fp4_asm_gemm: bool = False,
        out_dtype: torch.dtype | None = torch.bfloat16,
        x_scales: torch.Tensor | None = None,
        rotation: torch.Tensor | None = None,
        rotation_size: int = 0,
    ) -> torch.Tensor:
        M = x.shape[0]
        N = weight.shape[0]
        K = weight.shape[1]
        # Fused rotation+quant dispatch (real E2E benchmark on MI355X):
        # Gluon Triton is optimal for ALL M (20-105% faster than Separated).
        # HIP MFMA is within 5% of Gluon at M=1 only; Gluon 7-42% faster for M≥2.
        # Default: always Gluon (HIP_FUSED_MAX_M=0). Set VLLM_HIP_FUSED_MAX_M=N
        # to use HIP for M<N (e.g., =2 for M=1 decode only, =9999 for always HIP).
        if rotation is not None and rotation_size > 0 and _has_fused_triton_rot_quant:
            x_2d = x.reshape(-1, x.shape[-1])
            K_in = x.shape[-1]
            is_tuned_decode = (
                rocm_use_aiter_fp4_asm_gemm
                and M < DECODE_MAX_M
                and rocm_aiter_ops.is_triton_gemm_afp4wfp4_presh_ws_tuned(N, K)
            )
            shuffle_scales = not is_tuned_decode
            # Pick kernel: default Gluon (optimal for all M); HIP if M<HIP_FUSED_MAX_M or Gluon unavailable
            use_hip = _has_hip_mfma and (M < HIP_FUSED_MAX_M or not _has_gluon)
            # Cached pre-allocated buffers — eliminates per-call torch.empty + zero_()
            fp4_buf, sc_buf = _get_gemm_rot_quant_bufs(
                M, K_in, is_tuned_decode, x.device)
            if use_hip:
                # No sc_buf.zero_() per call — kernel only writes live cells,
                # padding cells stay zero from one-time init.
                _hip_mfma_rot_quant_dense(
                    x_2d, rotation, fp4_buf, sc_buf,
                    rotation_size, shuffle_scales)
                x_q = fp4_buf
                x_s = sc_buf
            else:
                x_q, x_s = _fused_rot_quant_gluon(
                    x_2d, rotation, rotation_size,
                    fp4_out=fp4_buf, scales_out=sc_buf,
                    shuffle_scales=shuffle_scales,
                )
            x_q = x_q.view(torch.float4_e2m1fn_x2)
            x_s = x_s.view(torch.float8_e8m0fnu)
            x_scales = x_s
            x = x_q
        if rocm_use_aiter_fp4_asm_gemm:
            if M <= 64 and rocm_aiter_ops.is_triton_gemm_afp4wfp4_presh_ws_tuned(N, K):
                if x_scales is None:
                    # Tuned decode M < 32 needs raw scales; else shuffled
                    shuffle_hip = M >= DECODE_MAX_M
                    x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=shuffle_hip)
                else:
                    x_q = x
                    x_s = x_scales

                if M >= 32:
                    x_s = x_s.view(torch.uint8).view(x_s.shape[0] // 32, -1)
                else:
                    x_s = x_s[:M, ...].view(torch.uint8)

                y = torch.empty(M, N, device=x_q.device, dtype=out_dtype)
                gemm_afp4wfp4_preshuffled_weight_scales(
                    x_q.view(torch.uint8),
                    weight.view(torch.uint8).view(weight.shape[0] // 16, -1),
                    x_s,
                    weight_scale.view(torch.uint8).view(
                        weight_scale.shape[0] // 32, -1
                    ),
                    out_dtype,
                    y,
                )
            else:
                if x_scales is None:
                    shuffle_hip = True  # untuned path always needs shuffled scales
                    x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=shuffle_hip)
                else:
                    x_q = x
                    x_s = x_scales

                # 32 alignment is enough for dim0 padding of output for
                # gemm_a4w4 kernel
                y = torch.empty(
                    (M + 31) // 32 * 32,
                    weight.shape[0],
                    device=x_q.device,
                    dtype=out_dtype,
                )

                w = weight.view(torch.float4_e2m1fn_x2) if weight.dtype == torch.uint8 else weight
                ws = weight_scale.view(torch.float8_e8m0fnu) if weight_scale.dtype == torch.uint8 else weight_scale
                gemm_a4w4(
                    x_q, w, x_s, ws.view(x_s.dtype), y, bpreshuffle=True
                )
            return y[:M]
        else:
            if x_scales is None:
                x_q, x_s = dynamic_mxfp4_quant(x)
            else:
                x_q = x
                x_s = x_scales
            y = torch.empty(
                x_q.shape[0], weight.shape[0], device=x_q.device, dtype=out_dtype
            )

            gemm_afp4wfp4(x_q, weight, x_s, weight_scale.T, out_dtype, y)
            return y

    def gemm_with_dynamic_quant_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        rocm_use_aiter_fp4_asm_gemm: bool = False,
        out_dtype: torch.dtype | None = torch.bfloat16,
        x_scales: torch.Tensor | None = None,
        rotation: torch.Tensor | None = None,
        rotation_size: int = 0,
    ) -> torch.Tensor:
        return torch.empty(
            (*x.shape[:-1], weight.shape[0]), dtype=out_dtype, device=x.device
        )

    direct_register_custom_op(
        op_name="gemm_with_dynamic_quant",
        op_func=gemm_with_dynamic_quant,
        mutates_args=[],
        fake_impl=gemm_with_dynamic_quant_fake,
        dispatch_key=current_platform.dispatch_key,
    )
except (ImportError, AttributeError, RuntimeError):
    if current_platform.is_rocm():
        logger.warning(
            "AITER is not found or QuarkOCP_MX is not supported on the current "
            "platform. QuarkOCP_MX quantization will not be available."
        )
    dynamic_mxfp4_quant = gemm_afp4wfp4 = None


class QuarkOCP_MX(QuarkScheme):
    def __init__(
        self,
        weight_quant_spec: dict[str, Any],
        input_quant_spec: dict[str, Any],
        quant_config: dict[str, Any],
        layer_names: list[str],
    ):
        self.out_dtype = torch.get_default_dtype()
        self.qscheme = "per_group"
        self.weight_quant_spec = weight_quant_spec
        self.input_quant_spec = input_quant_spec

        (
            self.use_online_rotation,
            self.rotation_config,
            self.rotation_size,
        ) = OrthogonalTransform.setup_transform(
            quant_config=quant_config, layer_names=layer_names
        )
        # Hadamard: trainable=False, matrix generated in vLLM; trained: trainable=True, matrix loaded from checkpoint
        self.is_hadamard_rotation = bool(
            self.use_online_rotation
            and self.rotation_config is not None
            and not self.rotation_config.get("trainable", True)
        )
        self.weight_dtype = weight_quant_spec["dtype"].replace("fp", "mxfp")
        self.input_dtype = input_quant_spec["dtype"].replace("fp", "mxfp")

        self.ocp_mx_scheme = OCP_MX_Scheme.from_quant_dtype(
            self.input_dtype, self.weight_dtype
        )

        if self.weight_dtype == "mxfp4":
            self.packed_factor: int | Fraction = 2
            self.dequant_func = dequant_mxfp4
        else:
            self.packed_factor = Fraction(numerator=8, denominator=6)
            self.dequant_func = partial(
                dequant_mxfp6, quant_dtype=self.weight_dtype.replace("mx", "")
            )

        if self.input_dtype == "mxfp4":
            self.quant_dequant_func = quant_dequant_mxfp4
        else:
            self.quant_dequant_func = partial(
                quant_dequant_mxfp6, quant_dtype=self.input_dtype.replace("mx", "")
            )

        self.static_input_scales = not input_quant_spec.get("is_dynamic")

        if self.static_input_scales:
            raise NotImplementedError(
                "QuarkOCP_MX with static input scales is currently not "
                "implemented. Please open an issue."
            )

        # TODO: integrate (or test) mixed-precision kernel.
        self.emulate = not current_platform.supports_mx() or (
            self.input_dtype != "mxfp4" or self.weight_dtype != "mxfp4"
        )

        self.rocm_use_aiter_fp4_asm_gemm = is_rocm_aiter_fp4_asm_gemm_enabled()

        # Fused rotation+quant vs separated: controlled by env switch
        self.use_fused_rotation_quant = (
            self.use_online_rotation
            and _has_fused_triton_rot_quant
            and not self.emulate
            and is_fused_rotation_quant_enabled()
        )
        if self.use_fused_rotation_quant:
            logger.info("Using fused Triton rotation+MXFP4 quant kernel")
        elif self.use_online_rotation and _has_fused_triton_rot_quant and not self.emulate:
            logger.info(
                "Using separated rotation+quant (set VLLM_USE_FUSED_ROTATION_QUANT=1 for fused)"
            )

        if not self.emulate and (dynamic_mxfp4_quant is None or gemm_afp4wfp4 is None):
            # Currently need these kernels if not emulating
            raise NotImplementedError(
                f"{self.__class__.__name__} requires AITER to be installed "
                "for non-emulation mode! Please refer to "
                "https://github.com/ROCm/aiter for installation details."
            )

        if not current_platform.supports_mx():
            logger.warning_once(
                "The current platform does not support native MXFP4/MXFP6 "
                "computation. Simulated weight dequantization and activation "
                "QDQ (quantize and dequantize) will be used, with the linear "
                "layers computed in high precision."
            )

        if current_platform.supports_mx() and (
            self.input_dtype != "mxfp4" or self.weight_dtype != "mxfp4"
        ):
            logger.warning_once(
                "The current platform supports native MXFP4/MXFP6 "
                f"computation, but kernels for input_dtype={self.input_dtype} "
                f"and weight_dtype={self.weight_dtype} are not yet integrated "
                "in vLLM. Simulated weight dequantization and activation "
                "QDQ (quantize and dequantize) will be used, with the linear "
                "layers computed in high precision."
            )

    def get_packed_dim(self, dim: int, quant_dtype: str):
        if quant_dtype == "mxfp4":
            assert dim % 2 == 0
            return dim // 2
        elif quant_dtype in {"mxfp6_e3m2", "mxfp6_e2m3"}:
            # FP6 packs 4 * 6 = 24 bits on 3 bytes.
            assert (dim * 3) % 4 == 0
            return (dim * 3) // 4
        else:
            raise NotImplementedError(
                "Unsupported quant_dtype in QuarkOCP_MX.get_packed_dim, "
                f"got quant_dtype={quant_dtype}. Something is wrong, please "
                "open an issue."
            )

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)

        if self.emulate:
            layer.weight_scale = torch.nn.Parameter(
                layer.weight_scale.data, requires_grad=False
            )
        else:
            if self.rocm_use_aiter_fp4_asm_gemm:
                # shuffle weight scale
                weight_scale_shuffle = layer.weight_scale.data
                sm, sn = weight_scale_shuffle.shape
                weight_scale_shuffle = weight_scale_shuffle.view(
                    sm // 32, 2, 16, sn // 8, 2, 4, 1
                )
                weight_scale_shuffle = weight_scale_shuffle.permute(
                    0, 3, 5, 2, 4, 1, 6
                ).contiguous()
                weight_scale_shuffle = weight_scale_shuffle.view(sm, sn)
                layer.weight_scale = torch.nn.Parameter(
                    weight_scale_shuffle, requires_grad=False
                )

                # shuffle weight
                weight_shuffle = layer.weight.data
                weight_shuffle = shuffle_weight(weight_shuffle, layout=(16, 16))
                layer.weight = torch.nn.Parameter(weight_shuffle, requires_grad=False)
            else:
                layer.weight_scale = torch.nn.Parameter(
                    layer.weight_scale.data.T.contiguous(), requires_grad=False
                )

        if self.use_online_rotation:
            self.input_transform.post_process_transform()

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                self.get_packed_dim(input_size_per_partition, self.weight_dtype),
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=self.packed_factor,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // OCP_MX_BLOCK_SIZE,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        if self.use_online_rotation:
            dtype = torch.float64 if self.rotation_config["trainable"] else torch.int8  # type: ignore[index]
            # Hadamard (trainable=False): Quark does not export the matrix; generate in vLLM for both fused and separated
            if self.is_hadamard_rotation:
                rot_data = build_hadamard_matrix(self.rotation_size, dtype=dtype)
            else:
                rot_data = torch.empty(self.rotation_size, self.rotation_size, dtype=dtype)
            input_rotation = ModelWeightParameter(
                data=rot_data,
                input_dim=1,
                output_dim=0,
                weight_loader=rotation_weight_loader,
            )
            layer.register_parameter("input_rotation", input_rotation)

            self.input_transform = OrthogonalTransform(
                layer.input_rotation, self.rotation_config
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_fused_rotation_quant:
            # Fused rotation+quant inside gemm_with_dynamic_quant (CUDAGraph safe)
            rotation = self.input_transform.input_rotation.data
            return torch.ops.vllm.gemm_with_dynamic_quant(
                x,
                layer.weight,
                layer.weight_scale,
                self.rocm_use_aiter_fp4_asm_gemm,
                self.out_dtype,
                rotation=rotation,
                rotation_size=self.rotation_size,
            )
        elif self.use_online_rotation:
            # Separated path: rotation matmul + separate quant
            x = self.input_transform(x)

        if self.emulate:
            dq_w = self.dequant_func(layer.weight, layer.weight_scale, x.dtype)
            qdq_x = self.quant_dequant_func(x)
            return F.linear(qdq_x, dq_w, bias)
        else:
            return torch.ops.vllm.gemm_with_dynamic_quant(
                x,
                layer.weight,
                layer.weight_scale,
                self.rocm_use_aiter_fp4_asm_gemm,
                self.out_dtype,
            )
