#!/usr/bin/env python3
"""
Test + Benchmark for HIP MoE 3-in-1 kernel: Fused Rotation + MXFP4 Quant + Sorted Scale Scatter.

Validates correctness against reference (gluon_kw8 + moe_mxfp4_sort).
Measures latency for M=1,2,4,8,16,32.

Usage:
    HIP_VISIBLE_DEVICES=0 python3 test_hip_moe_3in1.py

Compile the kernel first:
    cd /data/jiangyon/vllm_rotation/hip_rotation_quant
    hipcc --offload-arch=gfx950 -shared -fPIC -O3 \
      -o fused_rot_quant_sort_m_gt1.so fused_rot_quant_sort_m_gt1.hip
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
from vllm.model_executor.layers.quantization.quark.fused_rotation_quant_gluon_v2_kw8 import (
    fused_gluon_v2_kw8,
)

HIP_DIR = "/data/jiangyon/vllm_rotation/hip_rotation_quant"
SO_PATH = f"{HIP_DIR}/fused_rot_quant_sort_m_gt1.so"
HIP_PATH = f"{HIP_DIR}/fused_rot_quant_sort_m_gt1.hip"

E, TOPK, K, RS, QG = 128, 8, 2048, 128, 32
N_I = K // QG  # 64
DEVICE = "cuda"


def compile_kernel():
    """Compile the HIP kernel."""
    print(f"Compiling {HIP_PATH} ...")
    ret = os.system(
        f"hipcc --offload-arch=gfx950 -shared -fPIC -O3 -o {SO_PATH} {HIP_PATH} 2>&1"
    )
    if ret != 0:
        print("COMPILATION FAILED")
        sys.exit(1)
    print("Compilation OK")


def load_kernel():
    """Load the compiled .so via ctypes."""
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


def reference_2kernel(x, rotation, sorted_ids, num_valid_ids, M):
    """Reference: gluon_kw8 + moe_mxfp4_sort."""
    fp4 = torch.empty((M, K // 2), dtype=torch.uint8, device=DEVICE)
    raw_sc = torch.empty((M, N_I), dtype=torch.uint8, device=DEVICE)
    fp4, raw_sc = fused_gluon_v2_kw8(
        x, rotation, RS, fp4_out=fp4, scales_out=raw_sc, shuffle_scales=False
    )
    sorted_sc = moe_mxfp4_sort(
        raw_sc, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=M, block_size=32,
    )
    return fp4.view(torch.uint8), sorted_sc.view(torch.uint8).reshape(-1, N_I)


def hip_3in1(lib, x, rotation, sorted_ids, num_valid_ids, M, m_o, m_pad):
    """HIP 3-in-1 kernel."""
    fp4 = torch.empty((M, K // 2), dtype=torch.uint8, device=DEVICE)
    sc = torch.zeros((m_pad, N_I), dtype=torch.uint8, device=DEVICE)
    lib.launch_fused_rot_quant_sort_m_gt1(
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
    print("Correctness Test: HIP 3-in-1 vs Reference (gluon_kw8 + moe_mxfp4_sort)")
    print("=" * 70)

    all_pass = True
    for M in M_values:
        x, rot, si, nv, m_o, m_pad = build_inputs(M)
        num_valid = int(nv[0])
        valid_rows = ((num_valid + 31) // 32) * 32

        ref_fp4, ref_sc = reference_2kernel(x, rot, si, nv, M)
        hip_fp4, hip_sc = hip_3in1(lib, x, rot, si, nv, M, m_o, m_pad)
        torch.cuda.synchronize()

        fp4_match = (ref_fp4 == hip_fp4).float().mean().item()

        ref_sc_valid = ref_sc[:valid_rows]
        hip_sc_valid = hip_sc[:valid_rows]
        min_rows = min(ref_sc_valid.shape[0], hip_sc_valid.shape[0])
        if min_rows > 0:
            scale_match = (ref_sc_valid[:min_rows] == hip_sc_valid[:min_rows]).float().mean().item()
        else:
            scale_match = 1.0

        ok = fp4_match > 0.99 and scale_match > 0.99
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
    print("Benchmark: HIP 3-in-1 vs 2-kernel Baseline")
    print("=" * 70)

    for M in M_values:
        x, rot, si, nv, m_o, m_pad = build_inputs(M)

        fp4_ref = torch.empty((M, K // 2), dtype=torch.uint8, device=DEVICE)
        sc_ref = torch.empty((M, N_I), dtype=torch.uint8, device=DEVICE)

        for _ in range(warmup):
            fused_gluon_v2_kw8(x, rot, RS, fp4_out=fp4_ref, scales_out=sc_ref, shuffle_scales=False)
            moe_mxfp4_sort(sc_ref, sorted_ids=si, num_valid_ids=nv, token_num=M, block_size=32)
        for _ in range(warmup):
            hip_3in1(lib, x, rot, si, nv, M, m_o, m_pad)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(repeat):
            fused_gluon_v2_kw8(x, rot, RS, fp4_out=fp4_ref, scales_out=sc_ref, shuffle_scales=False)
            moe_mxfp4_sort(sc_ref, sorted_ids=si, num_valid_ids=nv, token_num=M, block_size=32)
        torch.cuda.synchronize()
        ref_us = (time.perf_counter() - t0) / repeat * 1e6

        t0 = time.perf_counter()
        for _ in range(repeat):
            hip_3in1(lib, x, rot, si, nv, M, m_o, m_pad)
        torch.cuda.synchronize()
        hip_us = (time.perf_counter() - t0) / repeat * 1e6

        speedup = ref_us / hip_us if hip_us > 0 else 0
        print(
            f"  M={M:3d}: 2-kernel={ref_us:.1f}µs  HIP-3in1={hip_us:.1f}µs  "
            f"speedup={speedup:.2f}x"
        )


if __name__ == "__main__":
    compile_kernel()
    lib = load_kernel()
    ok = test_correctness(lib)
    benchmark(lib)
    if not ok:
        print("\nWARNING: Correctness tests failed! Fix before deploying.")
        sys.exit(1)
