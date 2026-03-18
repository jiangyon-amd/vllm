#!/usr/bin/env python3
"""
Standalone MFMA 16x16x32 BF16 correctness test.

Tests:
1. Identity matmul (A @ I = A)
2. Known values matmul
3. Random matmul vs PyTorch reference
4. Multi-K-tile matmul (K > 32)
"""
import torch
import ctypes
import subprocess
import os
import sys

DIR = "/data/jiangyon/vllm_rotation/hip_rotation_quant"
HIP_FILE = os.path.join(DIR, "test_standalone_mfma.hip")
SO_FILE = os.path.join(DIR, "test_standalone_mfma.so")


def compile():
    if os.path.exists(SO_FILE) and os.path.getmtime(SO_FILE) > os.path.getmtime(HIP_FILE):
        print("  Already compiled")
        return
    print("  Compiling...")
    result = subprocess.run(
        ["hipcc", "-shared", "-fPIC", "-O2", "--offload-arch=gfx950",
         "-o", SO_FILE, HIP_FILE],
        capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  COMPILE ERROR:\n{result.stderr}")
        sys.exit(1)
    print("  OK")


def run_mfma(lib, A, B, M, N, K):
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    lib.launch_mfma_16x16(
        ctypes.c_void_p(C.data_ptr()),
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_void_p(B.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K),
        ctypes.c_void_p(0))
    torch.cuda.synchronize()
    return C


def run_scalar(lib, A, B, M, N, K):
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    lib.launch_scalar_matmul(
        ctypes.c_void_p(C.data_ptr()),
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_void_p(B.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K),
        ctypes.c_void_p(0))
    torch.cuda.synchronize()
    return C


def test_identity(lib):
    """Test 1: A @ I = A"""
    print("\n=== Test 1: Identity (A @ I = A) ===")
    M, N, K = 16, 16, 32
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.zeros(K, N, dtype=torch.bfloat16, device="cuda")
    for i in range(min(K, N)):
        B[i, i] = 1.0

    C_mfma = run_mfma(lib, A, B, M, N, K)
    # Reference: first 16 columns of A (since N=16, we only see A[:, :16])
    C_ref = A[:, :16].float()

    diff = (C_mfma - C_ref).abs()
    max_diff = diff.max().item()
    match = max_diff < 0.01

    print(f"  A[0, :8]      = {A[0, :8].float().tolist()}")
    print(f"  C_mfma[0, :8] = {C_mfma[0, :8].tolist()}")
    print(f"  C_ref[0, :8]  = {C_ref[0, :8].tolist()}")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Result: {'PASS' if match else 'FAIL'}")
    return match


def test_known_values(lib):
    """Test 2: Simple known values"""
    print("\n=== Test 2: Known values ===")
    M, N, K = 4, 4, 32
    A = torch.zeros(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.zeros(K, N, dtype=torch.bfloat16, device="cuda")

    # A = [[1,0,...], [0,1,...], [0,0,1,...], [0,0,0,1,...]]
    for i in range(M):
        A[i, i] = 1.0
    # B = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16], 0...]
    for i in range(4):
        for j in range(N):
            B[i, j] = float(i * N + j + 1)

    C_mfma = run_mfma(lib, A, B, M, N, K)
    C_expect = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ], dtype=torch.float32, device="cuda")

    diff = (C_mfma[:4, :4] - C_expect).abs()
    max_diff = diff.max().item()
    match = max_diff < 0.01

    print(f"  C_mfma[:4,:4] =")
    for r in range(4):
        print(f"    {C_mfma[r, :4].tolist()}")
    print(f"  Expected:")
    for r in range(4):
        print(f"    {C_expect[r].tolist()}")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Result: {'PASS' if match else 'FAIL'}")
    return match


def test_random(lib):
    """Test 3: Random matmul vs PyTorch"""
    print("\n=== Test 3: Random matmul (16x16, K=32) ===")
    M, N, K = 16, 16, 32
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")

    C_mfma = run_mfma(lib, A, B, M, N, K)
    C_ref = torch.matmul(A.float(), B.float())

    diff = (C_mfma - C_ref).abs()
    max_diff = diff.max().item()
    rel_err = (diff / (C_ref.abs() + 1e-6)).max().item()
    # BF16 matmul has rounding diffs - allow larger tolerance
    match = max_diff < 1.0

    print(f"  C_mfma[0, :6] = {C_mfma[0, :6].tolist()}")
    print(f"  C_ref[0, :6]  = {C_ref[0, :6].tolist()}")
    print(f"  Max abs diff: {max_diff:.6f}")
    print(f"  Max rel err:  {rel_err:.6f}")
    print(f"  Result: {'PASS' if match else 'FAIL'}")
    return match


def test_multi_k(lib):
    """Test 4: K > 32 (multiple MFMA K-tiles)"""
    print("\n=== Test 4: Multi-K-tile (16x16, K=128) ===")
    M, N, K = 16, 16, 128
    torch.manual_seed(123)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")

    C_mfma = run_mfma(lib, A, B, M, N, K)
    C_ref = torch.matmul(A.float(), B.float())

    diff = (C_mfma - C_ref).abs()
    max_diff = diff.max().item()
    rel_err = (diff / (C_ref.abs() + 1e-6)).max().item()
    match = max_diff < 2.0  # larger K = more accumulation error

    print(f"  C_mfma[0, :6] = {C_mfma[0, :6].tolist()}")
    print(f"  C_ref[0, :6]  = {C_ref[0, :6].tolist()}")
    print(f"  Max abs diff: {max_diff:.6f}")
    print(f"  Max rel err:  {rel_err:.6f}")
    print(f"  Result: {'PASS' if match else 'FAIL'}")
    return match


def test_mfma_vs_scalar(lib):
    """Test 5: MFMA vs HIP scalar kernel (same precision)"""
    print("\n=== Test 5: MFMA vs scalar HIP kernel ===")
    M, N, K = 16, 16, 32
    torch.manual_seed(77)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")

    C_mfma = run_mfma(lib, A, B, M, N, K)
    C_scalar = run_scalar(lib, A, B, M, N, K)

    diff = (C_mfma - C_scalar).abs()
    max_diff = diff.max().item()
    match = max_diff < 0.5  # both use bf16→f32, but different accumulation order

    print(f"  C_mfma[0, :6]   = {C_mfma[0, :6].tolist()}")
    print(f"  C_scalar[0, :6] = {C_scalar[0, :6].tolist()}")
    print(f"  Max diff (MFMA vs scalar): {max_diff:.6f}")
    print(f"  Note: difference is expected (different accumulation order)")
    print(f"  Result: {'PASS' if match else 'FAIL'}")
    return match


def test_layout_dump(lib):
    """Test 6: Verify MFMA output layout visually"""
    print("\n=== Test 6: MFMA output layout verification ===")
    M, N, K = 16, 16, 32
    # A[r,k] = r+1 for k=0, 0 otherwise → C[r,c] = (r+1) * B[0,c]
    A = torch.zeros(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.zeros(K, N, dtype=torch.bfloat16, device="cuda")
    for r in range(M):
        A[r, 0] = float(r + 1)
    for c in range(N):
        B[0, c] = float(c + 1)

    C_mfma = run_mfma(lib, A, B, M, N, K)
    # Expected: C[r,c] = (r+1)*(c+1)
    ok = True
    print(f"  Expected C[r,c] = (r+1)*(c+1)")
    for r in range(min(M, 4)):
        row = C_mfma[r, :4].tolist()
        expected = [(r + 1) * (c + 1) for c in range(4)]
        row_ok = all(abs(row[i] - expected[i]) < 0.01 for i in range(4))
        status = "✓" if row_ok else "✗"
        print(f"  Row {r}: MFMA={row} expected={expected} {status}")
        if not row_ok:
            ok = False

    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("=" * 60)
    print("Standalone MFMA 16x16x32 BF16 Correctness Test")
    print("=" * 60)

    compile()
    lib = ctypes.CDLL(SO_FILE)

    results = []
    results.append(("Identity", test_identity(lib)))
    results.append(("Known values", test_known_values(lib)))
    results.append(("Random", test_random(lib)))
    results.append(("Multi-K-tile", test_multi_k(lib)))
    results.append(("MFMA vs scalar", test_mfma_vs_scalar(lib)))
    results.append(("Layout verification", test_layout_dump(lib)))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:25s} {status}")
        if not passed:
            all_pass = False

    print(f"\n{'All tests passed!' if all_pass else 'Some tests FAILED!'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
