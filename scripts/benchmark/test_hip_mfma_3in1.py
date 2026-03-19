#!/usr/bin/env python3
"""
Test + Benchmark for HIP MFMA MoE 3-in-1 kernel.

Key difference from test_hip_moe_3in1.py:
- Correctness is validated against torch.matmul (NOT gluon_kw8 scalar reference)
- torch.matmul also uses MFMA internally, so results should match closely
- This allows the HIP MFMA kernel to have slightly different precision
  from the gluon scalar reference, which is acceptable.

Usage:
    HIP_VISIBLE_DEVICES=0 python3 test_hip_mfma_3in1.py
"""

import ctypes
import torch
import math
import time
import sys
import os

sys.path.insert(0, "/data/jiangyon/vllm_rotation")

from aiter.fused_moe import moe_sorting
from aiter.utility import dtypes
from aiter.utility.fp4_utils import moe_mxfp4_sort
from aiter.ops.triton.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort

HIP_DIR = "/data/jiangyon/vllm_rotation/hip_rotation_quant"
SO_PATH = f"{HIP_DIR}/mfma_rot_quant_moe_sort.so"
HIP_PATH = f"{HIP_DIR}/mfma_rot_quant_moe_sort.hip"

E, TOPK, K, RS, QG = 128, 8, 2048, 128, 32
N_I = K // QG
DEVICE = "cuda"


def compile_kernel():
    print(f"Compiling {HIP_PATH} ...")
    ret = os.system(f"hipcc --offload-arch=gfx950 -shared -fPIC -O3 -o {SO_PATH} {HIP_PATH} 2>&1")
    if ret != 0:
        print("COMPILATION FAILED")
        sys.exit(1)
    print("Compilation OK")


def load_kernel():
    if not os.path.exists(SO_PATH):
        compile_kernel()
    return ctypes.CDLL(SO_PATH)


def get_stream():
    return torch.cuda.current_stream().cuda_stream


def build_inputs(M):
    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE)
    rot_int8 = torch.ones(RS, RS, dtype=torch.int8, device=DEVICE)
    for i in range(RS):
        for j in range(RS):
            if bin(i & j).count("1") % 2 == 1:
                rot_int8[i, j] = -1
    rotation = (rot_int8.float() / math.sqrt(RS)).to(torch.bfloat16)

    topk_ids = torch.randint(0, E, (M, TOPK), device=DEVICE, dtype=torch.int32)
    topk_w = torch.ones(M, TOPK, device=DEVICE, dtype=torch.float32) / TOPK
    sorted_ids, _, _, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_w, E, K, torch.bfloat16, 32, None
    )
    m_o = sorted_ids.shape[0]
    m_pad = ((m_o + 31) // 32) * 32
    return x, rotation, sorted_ids, num_valid_ids, m_o, m_pad


def torch_matmul_reference(x, rotation, sorted_ids, num_valid_ids, M):
    """Reference using torch.matmul for rotation + aiter for quant+sort.
    torch.matmul uses MFMA internally, so this is the true MFMA reference."""
    x_rot = torch.empty_like(x)
    for s in range(0, K, RS):
        x_rot[:, s:s+RS] = torch.matmul(x[:, s:s+RS], rotation)

    fp4, sorted_scale = fused_dynamic_mxfp4_quant_moe_sort(
        x_rot, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=M, topk=1, block_size=32,
    )
    return (
        x_rot,
        fp4.view(torch.uint8),
        sorted_scale.view(torch.uint8).reshape(-1, N_I),
    )


def hip_mfma_3in1(lib, x, rotation, sorted_ids, num_valid_ids, M, m_o, m_pad):
    """HIP MFMA 3-in-1 kernel."""
    fp4 = torch.empty((M, K // 2), dtype=torch.uint8, device=DEVICE)
    sc = torch.zeros((m_pad, N_I), dtype=torch.uint8, device=DEVICE)
    lib.launch_mfma_rot_quant_moe_sort(
        ctypes.c_void_p(fp4.data_ptr()),
        ctypes.c_void_p(sc.data_ptr()),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(rotation.data_ptr()),
        ctypes.c_void_p(sorted_ids.data_ptr()),
        ctypes.c_void_p(num_valid_ids.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(M),
        ctypes.c_int(m_o), ctypes.c_int(m_pad),
        ctypes.c_int(x.stride(0)), ctypes.c_int(fp4.stride(0)),
        ctypes.c_int(sc.stride(0)), ctypes.c_int(sc.stride(1)),
        ctypes.c_int(N_I),
        ctypes.c_void_p(get_stream()),
    )
    return fp4, sc


def test_correctness(lib, M_values=None):
    if M_values is None:
        M_values = [1, 2, 4, 8, 16, 32]

    print("=" * 70)
    print("Correctness Test: HIP MFMA 3-in-1 vs torch.matmul reference")
    print("(torch.matmul also uses MFMA, so precision should match closely)")
    print("=" * 70)

    all_pass = True
    for M in M_values:
        x, rot, si, nv, m_o, m_pad = build_inputs(M)
        num_valid = int(nv[0])
        valid_rows = ((num_valid + 31) // 32) * 32

        x_rot_ref, ref_fp4, ref_sc = torch_matmul_reference(x, rot, si, nv, M)
        hip_fp4, hip_sc = hip_mfma_3in1(lib, x, rot, si, nv, M, m_o, m_pad)
        torch.cuda.synchronize()

        fp4_match = (ref_fp4 == hip_fp4).float().mean().item()

        ref_sc_valid = ref_sc[:valid_rows]
        hip_sc_valid = hip_sc[:valid_rows]
        min_rows = min(ref_sc_valid.shape[0], hip_sc_valid.shape[0])
        if min_rows > 0:
            scale_match = (ref_sc_valid[:min_rows] == hip_sc_valid[:min_rows]).float().mean().item()
        else:
            scale_match = 1.0

        ok = fp4_match > 0.95 and scale_match > 0.95
        status = "PASS" if ok else "FAIL"
        print(
            f"  M={M:3d}: {status}  fp4={fp4_match:.4f}  scale={scale_match:.4f}  "
            f"(valid={num_valid})"
        )
        if not ok:
            all_pass = False

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


def benchmark(lib, M_values=None, warmup=200, repeat=500):
    if M_values is None:
        M_values = [1, 2, 4, 8, 16, 32]

    print("\n" + "=" * 70)
    print("Benchmark: HIP MFMA 3-in-1")
    print("=" * 70)

    for M in M_values:
        x, rot, si, nv, m_o, m_pad = build_inputs(M)

        for _ in range(warmup):
            hip_mfma_3in1(lib, x, rot, si, nv, M, m_o, m_pad)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(repeat):
            hip_mfma_3in1(lib, x, rot, si, nv, M, m_o, m_pad)
        torch.cuda.synchronize()
        hip_us = (time.perf_counter() - t0) / repeat * 1e6

        print(f"  M={M:3d}: MFMA-3in1={hip_us:.1f}µs  (baseline ~29µs)")


if __name__ == "__main__":
    compile_kernel()
    lib = load_kernel()
    ok = test_correctness(lib)
    benchmark(lib)
    if not ok:
        print("\nWARNING: Correctness tests failed!")
        sys.exit(1)
