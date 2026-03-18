"""HIP MFMA 3-in-1 kernel wrapper: Fused Rotation + MXFP4 Quant + MoE Sort.

Uses v_mfma_f32_16x16x32_bf16 + ds_read_tr + v_cvt_scalef32_pk_fp4_f32.
Single kernel for all M values (decode + prefill), ~12-15µs on MI355X.
"""

import ctypes
import logging
import os
import subprocess
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

_lib = None
_compile_attempted = False

HIP_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "hip_rotation_quant"
HIP_SRC = HIP_DIR / "mfma_rot_quant_moe_sort.hip"
HIP_SO = HIP_DIR / "mfma_rot_quant_moe_sort.so"


def _compile_kernel():
    global _compile_attempted
    if _compile_attempted:
        return HIP_SO.exists()
    _compile_attempted = True
    if not HIP_SRC.exists():
        logger.warning("HIP MFMA kernel source not found: %s", HIP_SRC)
        return False
    try:
        subprocess.run(
            ["hipcc", "--offload-arch=gfx950", "-shared", "-fPIC", "-O3",
             "-o", str(HIP_SO), str(HIP_SRC)],
            check=True, capture_output=True, text=True,
        )
        logger.info("Compiled HIP MFMA kernel: %s", HIP_SO)
        return True
    except Exception as e:
        logger.warning("Failed to compile HIP MFMA kernel: %s", e)
        return False


def _get_lib():
    global _lib
    if _lib is not None:
        return _lib
    if not HIP_SO.exists():
        if not _compile_kernel():
            return None
    try:
        _lib = ctypes.CDLL(str(HIP_SO))
        logger.info("Loaded HIP MFMA kernel: %s", HIP_SO)
        return _lib
    except Exception as e:
        logger.warning("Failed to load HIP MFMA kernel: %s", e)
        return None


def is_available() -> bool:
    return _get_lib() is not None


def fused_mfma_rot_quant_moe_sort(
    x: torch.Tensor,
    rotation: torch.Tensor,
    rotation_size: int,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    topk: int,
    block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    lib = _get_lib()
    assert lib is not None, "HIP MFMA kernel not available"

    M, K = x.shape
    RS = rotation_size
    QG = 32
    n_i = K // QG
    m_o = sorted_ids.shape[0]
    m_pad = ((m_o + block_size - 1) // block_size) * block_size

    fp4_out = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
    scale_out = torch.zeros((m_pad, n_i), dtype=torch.uint8, device=x.device)

    stream = torch.cuda.current_stream().cuda_stream

    lib.launch_mfma_rot_quant_moe_sort(
        ctypes.c_void_p(fp4_out.data_ptr()),
        ctypes.c_void_p(scale_out.data_ptr()),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(rotation.data_ptr()),
        ctypes.c_void_p(sorted_ids.data_ptr()),
        ctypes.c_void_p(num_valid_ids.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(token_num),
        ctypes.c_int(m_o), ctypes.c_int(m_pad),
        ctypes.c_int(x.stride(0)), ctypes.c_int(fp4_out.stride(0)),
        ctypes.c_int(scale_out.stride(0)), ctypes.c_int(scale_out.stride(1)),
        ctypes.c_int(n_i),
        ctypes.c_void_p(stream),
    )

    from aiter.utility import dtypes
    return fp4_out.view(dtypes.fp4x2), scale_out.view(dtypes.fp8_e8m0)
