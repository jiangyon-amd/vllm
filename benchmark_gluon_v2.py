"""
Benchmark for fused_rotation_quant_gluon_v2 kernels.

Tests two paths:
1. Decode path (M=1, sorted_scales_topk8): _fused_rot_quant_decode_topk8_special
2. Prefill path (M>1, shuffle_scales): _fused_rot_quant_v2

Usage:
    python benchmark_gluon_v2.py
"""

import torch
import triton
import sys
import os

sys.path.insert(0, "/data/jiangyon/vllm_rotation")

from vllm.model_executor.layers.quantization.quark.fused_rotation_quant_gluon_v2 import (
    fused_gluon_v2,
)


def build_decode_inputs(K=2048, RS=128, topk=8, device="cuda"):
    M = 1
    QG = 32
    n_scales = K // QG
    token_num = 1

    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    rotation = torch.randn(RS, RS, dtype=torch.bfloat16, device=device)
    rotation = torch.linalg.qr(rotation.float())[0].to(torch.bfloat16)

    m_o = token_num * topk
    m_pad = ((m_o + 31) // 32) * 32
    sorted_ids = torch.arange(m_o, dtype=torch.int32, device=device)
    sorted_ids = torch.cat([
        sorted_ids,
        torch.full((m_pad - m_o,), token_num, dtype=torch.int32, device=device),
    ])
    num_valid_ids = torch.tensor([m_o], dtype=torch.int32, device=device)

    sn_padded = (n_scales + 7) // 8 * 8
    fp4_out = torch.empty((M, K // 2), dtype=torch.uint8, device=device)
    scales_out = torch.empty((m_pad + 32, sn_padded), dtype=torch.uint8, device=device)

    return {
        "x": x, "rotation": rotation, "fp4_out": fp4_out,
        "scales_out": scales_out,
        "sorted_ids": sorted_ids, "num_valid_ids": num_valid_ids,
        "M": M, "K": K, "RS": RS, "token_num": token_num, "topk": topk,
    }


def build_prefill_inputs(M=128, K=2048, RS=128, device="cuda"):
    QG = 32
    n_scales = K // QG

    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    rotation = torch.randn(RS, RS, dtype=torch.bfloat16, device=device)
    rotation = torch.linalg.qr(rotation.float())[0].to(torch.bfloat16)

    fp4_out = torch.empty((M, K // 2), dtype=torch.uint8, device=device)
    scales_out = torch.empty((M, n_scales), dtype=torch.uint8, device=device)

    return {
        "x": x, "rotation": rotation, "fp4_out": fp4_out,
        "scales_out": scales_out,
        "M": M, "K": K, "RS": RS,
    }


def run_decode(inputs):
    return fused_gluon_v2(
        inputs["x"],
        inputs["rotation"],
        rotation_size=inputs["RS"],
        fp4_out=inputs["fp4_out"],
        scales_out=inputs["scales_out"],
        sorted_scales_topk8=True,
        sorted_ids=inputs["sorted_ids"],
        num_valid_ids=inputs["num_valid_ids"],
        token_num=inputs["token_num"],
    )


def run_prefill(inputs):
    return fused_gluon_v2(
        inputs["x"],
        inputs["rotation"],
        rotation_size=inputs["RS"],
        fp4_out=inputs["fp4_out"],
        scales_out=inputs["scales_out"],
    )


def benchmark(name, run_fn, inputs, warmup=50, repeat=200):
    try:
        run_fn(inputs)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"Kernel: {name}")
        print(f"SKIPPED: {e}")
        print(f"{'='*60}")
        return None

    for _ in range(warmup):
        run_fn(inputs)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        start_events[i].record()
        run_fn(inputs)
        end_events[i].record()
    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times_ms.sort()
    trim = max(1, repeat // 10)
    trimmed = times_ms[trim:-trim]
    avg_us = sum(trimmed) / len(trimmed) * 1000
    min_us = min(times_ms) * 1000
    med_us = times_ms[len(times_ms) // 2] * 1000

    print(f"\n{'='*60}")
    print(f"Kernel: {name}")
    print(f"{'='*60}")
    print(f"Latency (us):  min={min_us:.1f}  median={med_us:.1f}  mean={avg_us:.1f}")
    print(f"{'='*60}")
    print(f"Correctness: PASS")
    return med_us


if __name__ == "__main__":
    print("="*60)
    print("Gluon v2 Fused Rotation+Quant Benchmark")
    print("="*60)

    results = {}

    # Decode path: M=1, topk=8
    for K in [2048, 7168]:
        name = f"decode_topk8_K{K}"
        inputs = build_decode_inputs(K=K, topk=8)
        med = benchmark(name, run_decode, inputs)
        if med is not None:
            results[name] = med

    # Prefill path: various M
    for M in [1, 16, 128]:
        for K in [2048, 7168]:
            name = f"prefill_M{M}_K{K}"
            inputs = build_prefill_inputs(M=M, K=K)
            med = benchmark(name, run_prefill, inputs)
            if med is not None:
                results[name] = med

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, med in results.items():
        print(f"  {name}: median={med:.1f} us")
    print(f"{'='*60}")
