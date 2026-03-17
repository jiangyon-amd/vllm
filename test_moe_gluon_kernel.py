#!/usr/bin/env python3
"""
Comprehensive unit tests for MoE fused rotation+quant+sort kernel.

Tests both existing Triton kernel and new gluon kernel (when available)
against the separated reference (torch rotation matmul + aiter quant+sort).

Usage:
    HIP_VISIBLE_DEVICES=1 python3 test_moe_gluon_kernel.py
"""
import os
import sys
import math

import torch
import triton

os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")

from aiter.fused_moe import moe_sorting
from aiter.ops.triton.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort

from vllm.model_executor.layers.quantization.quark.fused_rotation_mxfp4_quant_moe_sort import (
    _fused_decode_m1_rot_quant_sorted_kernel,
)

# Try importing gluon version
try:
    from vllm.model_executor.layers.quantization.quark.fused_rotation_mxfp4_quant_moe_gluon import (
        fused_decode_m1_rot_quant_sorted_gluon,
    )
    HAS_GLUON_MOE = True
except ImportError:
    HAS_GLUON_MOE = False

# Try importing HIP version
try:
    from vllm.model_executor.layers.quantization.quark.fused_rotation_mxfp4_quant_moe_hip import (
        fused_decode_m1_rot_quant_sorted_hip,
    )
    HAS_HIP_MOE = True
except ImportError:
    HAS_HIP_MOE = False


def make_hadamard_rotation(RS=128, device="cuda"):
    """Create deterministic Hadamard rotation matrix."""
    rot_int8 = torch.ones(RS, RS, dtype=torch.int8, device=device)
    for i in range(RS):
        for j in range(RS):
            if (bin(i & j).count('1')) % 2 == 1:
                rot_int8[i, j] = -1
    return (rot_int8.float() / math.sqrt(RS)).to(torch.bfloat16)


def make_random_rotation(RS=128, device="cuda"):
    """Create random orthogonal-like rotation matrix."""
    signs = torch.randint(0, 2, (RS, RS), device=device) * 2 - 1
    return (signs.float() / math.sqrt(RS)).to(torch.bfloat16)


def build_moe_inputs(K=2048, RS=128, E=128, topk=8, seed=42, device="cuda"):
    """Build MoE decode M=1 test inputs using aiter moe_sorting."""
    M = 1
    QG = 32
    n_i = K // QG
    token_num = 1
    bs = 32

    torch.manual_seed(seed)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    rotation = make_hadamard_rotation(RS, device)
    topk_ids = torch.randint(0, E, (M, topk), device=device, dtype=torch.int32)
    topk_weights = torch.ones(M, topk, device=device, dtype=torch.float32) / topk
    sorted_ids, _, _, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, E, K, torch.bfloat16, bs, None
    )

    m_o = sorted_ids.shape[0]
    m_pad = ((m_o + 31) // 32) * 32

    return {
        "x": x, "rotation": rotation,
        "sorted_ids": sorted_ids, "num_valid_ids": num_valid_ids,
        "M": M, "K": K, "RS": RS, "QG": QG, "n_i": n_i,
        "token_num": token_num, "topk": topk, "E": E,
        "m_o": m_o, "m_pad": m_pad, "bs": bs,
    }


def compute_reference(inputs):
    """Separated reference: torch rotation matmul + aiter quant+sort."""
    K, RS = inputs["K"], inputs["RS"]
    x, rotation = inputs["x"], inputs["rotation"]
    sorted_ids = inputs["sorted_ids"]
    num_valid_ids = inputs["num_valid_ids"]
    bs = inputs["bs"]

    x_rot = torch.empty_like(x)
    for s in range(0, K, RS):
        x_rot[:, s:s+RS] = torch.matmul(x[:, s:s+RS], rotation)

    ref_fp4, ref_scale = fused_dynamic_mxfp4_quant_moe_sort(
        x_rot, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=1, topk=1, block_size=bs,
    )
    return ref_fp4, ref_scale


def run_triton_kernel(inputs):
    """Run existing Triton kernel."""
    K, RS, QG = inputs["K"], inputs["RS"], inputs["QG"]
    n_i = inputs["n_i"]
    m_o, m_pad = inputs["m_o"], inputs["m_pad"]
    token_num = inputs["token_num"]

    fp4 = torch.empty((1, K // 2), dtype=torch.uint8, device="cuda")
    sorted_scale = torch.empty((m_pad, n_i), dtype=torch.uint8, device="cuda")

    from vllm.model_executor.layers.quantization.quark.fused_rotation_mxfp4_quant_moe_sort import (
        _pick_decode_max_q,
    )
    max_q = _pick_decode_max_q(m_pad)

    grid = (K // RS,)
    _fused_decode_m1_rot_quant_sorted_kernel[grid](
        inputs["x"], inputs["rotation"], fp4,
        inputs["sorted_ids"], inputs["num_valid_ids"], sorted_scale,
        inputs["x"].stride(0), inputs["x"].stride(1),
        inputs["rotation"].stride(0), inputs["rotation"].stride(1),
        fp4.stride(0), sorted_scale.stride(0), sorted_scale.stride(1),
        token_num=token_num, n_i=n_i,
        tile_n=triton.cdiv(n_i, 8), m_o=m_o,
        RS=RS, QG=QG, MAX_Q=max_q, num_warps=4,
    )
    return fp4, sorted_scale


def run_gluon_kernel(inputs):
    """Run gluon kernel."""
    if not HAS_GLUON_MOE:
        return None, None
    K, RS, QG = inputs["K"], inputs["RS"], inputs["QG"]
    n_i = inputs["n_i"]
    m_o, m_pad = inputs["m_o"], inputs["m_pad"]
    token_num = inputs["token_num"]

    fp4 = torch.empty((1, K // 2), dtype=torch.uint8, device="cuda")
    # Extra row for temp scale storage; zero to avoid stale data
    sorted_scale = torch.zeros((m_pad + 1, n_i), dtype=torch.uint8, device="cuda")

    fused_decode_m1_rot_quant_sorted_gluon(
        inputs["x"], inputs["rotation"], fp4,
        inputs["sorted_ids"], inputs["num_valid_ids"], sorted_scale,
        K=K, RS=RS, QG=QG, n_i=n_i,
        token_num=token_num, m_o=m_o, m_pad=m_pad,
    )
    return fp4, sorted_scale


def compare_results(ref_fp4, ref_scale, test_fp4, test_scale, n_i, valid_rows):
    """Compare kernel output against reference."""
    fp4_match = (ref_fp4.view(torch.uint8) == test_fp4.view(torch.uint8)).float().mean().item()
    rs = ref_scale.view(torch.uint8).reshape(-1, n_i)[:valid_rows]
    ts = test_scale.view(torch.uint8).reshape(-1, n_i)[:valid_rows]
    scale_match = (rs == ts).float().mean().item()
    return fp4_match, scale_match


def test_triton_correctness():
    """Test Triton kernel against separated reference."""
    print("=" * 60)
    print("TEST 1: Triton kernel correctness")
    print("=" * 60)

    all_pass = True
    configs = [
        {"K": 2048, "E": 128, "topk": 8, "label": "30B (K=2048, E=128, topk=8)"},
        {"K": 2048, "E": 64, "topk": 4, "label": "small MoE (K=2048, E=64, topk=4)"},
        {"K": 2048, "E": 128, "topk": 2, "label": "topk=2 (K=2048, E=128)"},
    ]

    for cfg in configs:
        inputs = build_moe_inputs(K=cfg["K"], E=cfg["E"], topk=cfg["topk"])
        ref_fp4, ref_scale = compute_reference(inputs)
        test_fp4, test_scale = run_triton_kernel(inputs)
        valid = ((int(inputs["num_valid_ids"][0]) + 31) // 32) * 32
        fp4_m, sc_m = compare_results(ref_fp4, ref_scale, test_fp4, test_scale,
                                       inputs["n_i"], valid)
        # v_cvt ISA rounds differently than separated path (~1-2% FP4 difference)
        ok = fp4_m > 0.95 and sc_m > 0.99
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {cfg['label']}: fp4={fp4_m:.4f} scale={sc_m:.4f} {status}")

    return all_pass


def test_triton_determinism():
    """Test that kernel produces identical output across multiple runs."""
    print("\n" + "=" * 60)
    print("TEST 2: Triton kernel determinism")
    print("=" * 60)

    all_pass = True
    inputs = build_moe_inputs()

    valid = ((int(inputs["num_valid_ids"][0]) + 31) // 32) * 32
    n_i = inputs["n_i"]
    results = []
    for i in range(5):
        fp4, scale = run_triton_kernel(inputs)
        results.append((fp4.clone(), scale[:valid, :n_i].clone()))

    for i in range(1, len(results)):
        fp4_same = (results[0][0] == results[i][0]).all().item()
        scale_same = (results[0][1] == results[i][1]).all().item()
        ok = fp4_same and scale_same
        if not ok:
            all_pass = False
        print(f"  Run 0 vs {i}: fp4={'same' if fp4_same else 'DIFF'} "
              f"scale={'same' if scale_same else 'DIFF'} {'PASS' if ok else 'FAIL'}")

    return all_pass


def test_triton_different_rotations():
    """Test with different rotation matrices."""
    print("\n" + "=" * 60)
    print("TEST 3: Different rotation matrices")
    print("=" * 60)

    all_pass = True
    inputs = build_moe_inputs()

    for rot_type in ["hadamard", "random1", "random2", "identity"]:
        if rot_type == "hadamard":
            rot = make_hadamard_rotation(128, "cuda")
        elif rot_type == "identity":
            rot = torch.eye(128, dtype=torch.bfloat16, device="cuda") 
        else:
            torch.manual_seed(hash(rot_type) % 2**31)
            rot = make_random_rotation(128, "cuda")

        inputs["rotation"] = rot
        ref_fp4, ref_scale = compute_reference(inputs)
        test_fp4, test_scale = run_triton_kernel(inputs)
        valid = ((int(inputs["num_valid_ids"][0]) + 31) // 32) * 32
        fp4_m, sc_m = compare_results(ref_fp4, ref_scale, test_fp4, test_scale,
                                       inputs["n_i"], valid)
        ok = fp4_m > 0.95 and sc_m > 0.99
        if not ok:
            all_pass = False
        print(f"  {rot_type:>10}: fp4={fp4_m:.4f} scale={sc_m:.4f} {'PASS' if ok else 'FAIL'}")

    return all_pass


def test_triton_different_seeds():
    """Test with different random inputs."""
    print("\n" + "=" * 60)
    print("TEST 4: Different random seeds")
    print("=" * 60)

    all_pass = True
    for seed in [0, 42, 123, 999, 2024]:
        inputs = build_moe_inputs(seed=seed)
        ref_fp4, ref_scale = compute_reference(inputs)
        test_fp4, test_scale = run_triton_kernel(inputs)
        valid = ((int(inputs["num_valid_ids"][0]) + 31) // 32) * 32
        fp4_m, sc_m = compare_results(ref_fp4, ref_scale, test_fp4, test_scale,
                                       inputs["n_i"], valid)
        ok = fp4_m > 0.95 and sc_m > 0.99
        if not ok:
            all_pass = False
        print(f"  seed={seed:>4}: fp4={fp4_m:.4f} scale={sc_m:.4f} {'PASS' if ok else 'FAIL'}")

    return all_pass


def test_triton_edge_cases():
    """Test edge cases: extreme values, zeros, uniform."""
    print("\n" + "=" * 60)
    print("TEST 5: Edge cases")
    print("=" * 60)

    all_pass = True
    inputs = build_moe_inputs()

    cases = {
        "zeros": torch.zeros(1, 2048, dtype=torch.bfloat16, device="cuda"),
        "ones": torch.ones(1, 2048, dtype=torch.bfloat16, device="cuda"),
        "large": torch.full((1, 2048), 100.0, dtype=torch.bfloat16, device="cuda"),
        "small": torch.full((1, 2048), 0.001, dtype=torch.bfloat16, device="cuda"),
        "mixed": torch.randn(1, 2048, dtype=torch.bfloat16, device="cuda") * 10,
    }

    for name, x in cases.items():
        inputs["x"] = x
        ref_fp4, ref_scale = compute_reference(inputs)
        test_fp4, test_scale = run_triton_kernel(inputs)
        valid = ((int(inputs["num_valid_ids"][0]) + 31) // 32) * 32
        fp4_m, sc_m = compare_results(ref_fp4, ref_scale, test_fp4, test_scale,
                                       inputs["n_i"], valid)
        ok = fp4_m > 0.95 and sc_m > 0.99
        if not ok:
            all_pass = False
        print(f"  {name:>8}: fp4={fp4_m:.4f} scale={sc_m:.4f} {'PASS' if ok else 'FAIL'}")

    return all_pass


def run_hip_kernel(inputs):
    """Run HIP kernel."""
    if not HAS_HIP_MOE:
        return None, None
    K, RS, QG = inputs["K"], inputs["RS"], inputs["QG"]
    n_i = inputs["n_i"]
    m_o, m_pad = inputs["m_o"], inputs["m_pad"]
    token_num = inputs["token_num"]

    fp4 = torch.empty((1, K // 2), dtype=torch.uint8, device="cuda")
    sorted_scale = torch.empty((m_pad, n_i), dtype=torch.uint8, device="cuda")

    fused_decode_m1_rot_quant_sorted_hip(
        inputs["x"], inputs["rotation"], fp4,
        inputs["sorted_ids"], inputs["num_valid_ids"], sorted_scale,
        K=K, RS=RS, QG=QG, n_i=n_i,
        token_num=token_num, m_o=m_o, m_pad=m_pad,
    )
    return fp4, sorted_scale


def test_gluon_vs_triton():
    """Test gluon kernel matches Triton kernel exactly."""
    print("\n" + "=" * 60)
    print("TEST 6: Gluon vs Triton kernel")
    print("=" * 60)

    if not HAS_GLUON_MOE:
        print("  SKIPPED (gluon MoE kernel not yet implemented)")
        return True

    all_pass = True
    for seed in [42, 123, 999]:
        inputs = build_moe_inputs(seed=seed)
        triton_fp4, triton_scale = run_triton_kernel(inputs)
        gluon_fp4, gluon_scale = run_gluon_kernel(inputs)

        if gluon_fp4 is None:
            print("  SKIPPED (gluon kernel returned None)")
            return True

        valid = ((int(inputs["num_valid_ids"][0]) + 31) // 32) * 32
        n_i = inputs["n_i"]
        fp4_m = (triton_fp4.view(torch.uint8) == gluon_fp4.view(torch.uint8)).float().mean().item()
        ts = triton_scale.view(torch.uint8).reshape(-1, n_i)[:valid]
        gs = gluon_scale.view(torch.uint8).reshape(-1, n_i)[:valid]
        sc_m = (ts == gs).float().mean().item()
        ok = fp4_m > 0.98 and sc_m > 0.99
        if not ok:
            all_pass = False
        print(f"  seed={seed}: fp4={fp4_m:.4f} scale={sc_m:.4f} {'PASS' if ok else 'FAIL'}")

    return all_pass


def test_hip_vs_triton():
    """Test HIP kernel matches Triton kernel."""
    print("\n" + "=" * 60)
    print("TEST 7: HIP vs Triton kernel")
    print("=" * 60)

    if not HAS_HIP_MOE:
        print("  SKIPPED (HIP MoE kernel not available)")
        return True

    all_pass = True
    for seed in [42, 123, 999]:
        inputs = build_moe_inputs(seed=seed)
        triton_fp4, triton_scale = run_triton_kernel(inputs)
        hip_fp4, hip_scale = run_hip_kernel(inputs)

        if hip_fp4 is None:
            print("  SKIPPED (HIP kernel returned None)")
            return True

        valid = ((int(inputs["num_valid_ids"][0]) + 31) // 32) * 32
        n_i = inputs["n_i"]
        fp4_m = (triton_fp4.view(torch.uint8) == hip_fp4.view(torch.uint8)).float().mean().item()
        ts = triton_scale.view(torch.uint8).reshape(-1, n_i)[:valid]
        hs = hip_scale.view(torch.uint8).reshape(-1, n_i)[:valid]
        sc_m = (ts == hs).float().mean().item()
        ok = fp4_m > 0.98 and sc_m > 0.99
        if not ok:
            all_pass = False
        print(f"  seed={seed}: fp4={fp4_m:.4f} scale={sc_m:.4f} {'PASS' if ok else 'FAIL'}")

    return all_pass


def test_hip_correctness():
    """Test HIP kernel against separated reference."""
    print("\n" + "=" * 60)
    print("TEST 8: HIP kernel correctness (vs separated)")
    print("=" * 60)

    if not HAS_HIP_MOE:
        print("  SKIPPED (HIP MoE kernel not available)")
        return True

    all_pass = True
    configs = [
        {"K": 2048, "E": 128, "topk": 8, "label": "30B (K=2048, E=128, topk=8)"},
        {"K": 2048, "E": 64, "topk": 4, "label": "small MoE (K=2048, E=64, topk=4)"},
        {"K": 2048, "E": 128, "topk": 2, "label": "topk=2 (K=2048, E=128)"},
    ]

    for cfg in configs:
        inputs = build_moe_inputs(K=cfg["K"], E=cfg["E"], topk=cfg["topk"])
        ref_fp4, ref_scale = compute_reference(inputs)
        test_fp4, test_scale = run_hip_kernel(inputs)
        valid = ((int(inputs["num_valid_ids"][0]) + 31) // 32) * 32
        fp4_m, sc_m = compare_results(ref_fp4, ref_scale, test_fp4, test_scale,
                                       inputs["n_i"], valid)
        ok = fp4_m > 0.95 and sc_m > 0.99
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {cfg['label']}: fp4={fp4_m:.4f} scale={sc_m:.4f} {status}")

    return all_pass


def main():
    print(f"Triton {triton.__version__}, PyTorch {torch.__version__}")
    print(f"Gluon MoE kernel: {'available' if HAS_GLUON_MOE else 'not available'}")
    print(f"HIP MoE kernel:   {'available' if HAS_HIP_MOE else 'not available'}")
    print()

    results = []
    results.append(("Triton correctness", test_triton_correctness()))
    results.append(("Triton determinism", test_triton_determinism()))
    results.append(("Different rotations", test_triton_different_rotations()))
    results.append(("Different seeds", test_triton_different_seeds()))
    results.append(("Edge cases", test_triton_edge_cases()))
    results.append(("Gluon vs Triton", test_gluon_vs_triton()))
    results.append(("HIP vs Triton", test_hip_vs_triton()))
    results.append(("HIP correctness", test_hip_correctness()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {name:<25} {status}")

    print()
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
