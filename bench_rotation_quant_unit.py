#!/usr/bin/env python3
"""
Unit benchmark: rotation + MXFP4 quantization
Compares three implementations (precision-aligned to HIP quant):
  1. Baseline: torch rotation (bf16 matmul) + per_1x32_f4_quant_hip (separate)
  2. Triton v5: fused rotation + quant (standard Triton, hw FP4 instruction)
  3. Gluon v13: fused rotation + quant (Gluon, hw FP4 instruction, MFMA layout)
"""
import sys
sys.path.insert(0, '/opt/aiter')
sys.path.insert(0, '/data/jiangyon/vllm_rotation')

import torch
import time

device = "cuda:0"
RS = 128  # rotation_size
WARMUP = 500
N_ITER = 5000

# ============================================================
# Implementation 1a: Baseline — torch rotation + aiter triton quant
# ============================================================
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.quant import per_1x32_f4_quant_triton

def baseline_torch_aiter(x, rotation, rotation_size=128):
    """Two-step: torch bf16 matmul + aiter triton MXFP4 quant (dynamic_mxfp4_quant)."""
    M, K = x.shape
    x_r = x.reshape(M, K // rotation_size, rotation_size)
    x_rotated = torch.bmm(x_r, rotation.unsqueeze(0).expand(x_r.shape[0], -1, -1))
    x_rotated = x_rotated.reshape(M, K)
    fp4, scales = dynamic_mxfp4_quant(x_rotated)
    return fp4, scales

# ============================================================
# Implementation 1b: Baseline — torch rotation + HIP-compat triton quant
# ============================================================
def baseline_torch_hip_triton(x, rotation, rotation_size=128):
    """Two-step: torch bf16 matmul + per_1x32_f4_quant_triton (HIP-compatible)."""
    M, K = x.shape
    x_r = x.reshape(M, K // rotation_size, rotation_size)
    x_rotated = torch.bmm(x_r, rotation.unsqueeze(0).expand(x_r.shape[0], -1, -1))
    x_rotated = x_rotated.reshape(M, K)
    fp4, scales = per_1x32_f4_quant_triton(x_rotated)
    return fp4.view(torch.uint8), scales.view(torch.uint8)[:M]

# ============================================================
# Implementation 2: Triton v5 (fused, standard Triton)
# ============================================================
# v5 needs _mxfp4_quant_op, patch the import
import aiter.ops.triton._triton_kernels.quant as _quant_pkg
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
_quant_pkg._mxfp4_quant_op = _mxfp4_quant_op

from fused_rotation_mxfp4_quant_v5 import fused_rotation_mxfp4_quant as triton_v5

# ============================================================
# Implementation 3: Gluon v13 (fused, Gluon hw FP4)
# ============================================================
from fused_rotation_mxfp4_gluon_v13 import fused_gluon_v13 as gluon_v13


def bench_fn(fn, x, rot, rs, fp4_buf=None, sc_buf=None, label=""):
    """Benchmark a function, return avg time in microseconds."""
    # Warmup
    for _ in range(WARMUP):
        if fp4_buf is not None:
            _ = fn(x, rot, rs, fp4_out=fp4_buf, scales_out=sc_buf)
        else:
            _ = fn(x, rot, rs)
    torch.cuda.synchronize()

    # Timed
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        if fp4_buf is not None:
            _ = fn(x, rot, rs, fp4_out=fp4_buf, scales_out=sc_buf)
        else:
            _ = fn(x, rot, rs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_ITER * 1e6


def main():
    print("=" * 90)
    print("Rotation + MXFP4 Quantization — Unit Performance Benchmark")
    print("=" * 90)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Rotation size: {RS}, Warmup: {WARMUP}, Iterations: {N_ITER}")
    print(f"Implementations:")
    print(f"  1a. Base-aiter:  torch.bmm + dynamic_mxfp4_quant (2-step)")
    print(f"  1b. Base-HIP:    torch.bmm + per_1x32_f4_quant_triton (2-step, HIP-compat)")
    print(f"  2.  Triton v5:   fused kernel (standard Triton, hw FP4)")
    print(f"  3.  Gluon v13:   fused kernel (Gluon, hw FP4, MFMA layout quant)")
    print(f"  4.  Gluon v13p:  same as 3 with pre-allocated output")
    print()

    shapes = [
        (1, 5120),
        (4, 5120),
        (8, 5120),
        (16, 5120),
        (32, 5120),
        (64, 5120),
        (128, 5120),
        (256, 5120),
        (512, 5120),
        (1024, 5120),
        # gate_up_proj size
        (1, 27648),
        (32, 27648),
        (128, 27648),
    ]

    header = (f"{'Shape':>14s}  {'Base-aiter':>10s}  {'Base-HIP':>10s}  "
              f"{'Triton_v5':>10s}  {'Gluon_v13':>10s}  {'Gluon_pre':>10s}  "
              f"{'v5/hip':>7s}  {'glu/hip':>7s}  {'pre/hip':>7s}")
    print(header)
    print("-" * len(header))

    for M, K in shapes:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        rot = torch.linalg.qr(torch.randn(RS, RS, device=device))[0].to(torch.bfloat16)

        # Pre-allocate for gluon v13
        fp4_buf = torch.empty((M, K // 2), dtype=torch.uint8, device=device)
        sc_buf = torch.empty((M, K // 32), dtype=torch.uint8, device=device)

        # Benchmark each
        t_aiter = bench_fn(baseline_torch_aiter, x, rot, RS, label="base-aiter")
        t_hip = bench_fn(baseline_torch_hip_triton, x, rot, RS, label="base-hip")
        t_v5 = bench_fn(triton_v5, x, rot, RS, label="triton_v5")
        t_gluon = bench_fn(gluon_v13, x, rot, RS, label="gluon_v13")
        t_gluon_pre = bench_fn(gluon_v13, x, rot, RS,
                                fp4_buf=fp4_buf, sc_buf=sc_buf, label="gluon_pre")

        # Speedup vs HIP baseline
        sp_v5 = t_hip / t_v5 if t_v5 > 0 else 0
        sp_glu = t_hip / t_gluon if t_gluon > 0 else 0
        sp_pre = t_hip / t_gluon_pre if t_gluon_pre > 0 else 0

        print(f"  {M:>4d}x{K:<5d}  {t_aiter:>8.1f}us  {t_hip:>8.1f}us  "
              f"{t_v5:>8.1f}us  {t_gluon:>8.1f}us  {t_gluon_pre:>8.1f}us  "
              f"{sp_v5:>6.2f}x  {sp_glu:>6.2f}x  {sp_pre:>6.2f}x")

    # Correctness spot-check
    print()
    print("=" * 70)
    print("Correctness spot-check (M=64, K=5120, rotation=identity)")
    print("=" * 70)
    rot_I = torch.eye(RS, dtype=torch.bfloat16, device=device)
    torch.manual_seed(42)
    x = torch.randn(64, 5120, dtype=torch.bfloat16, device=device)

    fp4_hip, sc_hip = baseline_torch_hip_triton(x, rot_I, RS)
    fp4_aiter, sc_aiter = baseline_torch_aiter(x, rot_I, RS)
    fp4_v5, sc_v5 = triton_v5(x, rot_I, RS)
    fp4_glu, sc_glu = gluon_v13(x, rot_I, RS)

    print("  vs HIP baseline (per_1x32_f4_quant_triton):")
    for name, fp4, sc in [("aiter_quant", fp4_aiter, sc_aiter),
                           ("triton_v5", fp4_v5, sc_v5),
                           ("gluon_v13", fp4_glu, sc_glu)]:
        fp4_diff = (fp4 != fp4_hip).sum().item()
        sc_diff = (sc != sc_hip).sum().item()
        pct = fp4_diff / fp4_hip.numel() * 100
        status = "✅ MATCH" if fp4_diff == 0 and sc_diff == 0 else f"fp4_diff={fp4_diff} ({pct:.1f}%), sc_diff={sc_diff}"
        print(f"    {name:>12s}: {status}")


if __name__ == "__main__":
    main()
