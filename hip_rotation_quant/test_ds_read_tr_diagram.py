#!/usr/bin/env python3
"""
Verify DS_READ_B64_TR_B16 matches AMD diagram exactly.

LDS [4][16], value = row * 16 + col (0..63):
  Row 0: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15]
  Row 1: [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
  Row 2: [32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47]
  Row 3: [48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]

Expected after transpose:
  t0  → col 0 from all rows: [ 0, 16, 32, 48]
  t1  → col 1:               [ 1, 17, 33, 49]
  t15 → col 15:              [15, 31, 47, 63]
"""
import torch, ctypes, subprocess, os, sys

DIR = "/data/jiangyon/vllm_rotation/hip_rotation_quant"
HIP = os.path.join(DIR, "test_ds_read_tr_diagram.hip")
SO = os.path.join(DIR, "test_ds_read_tr_diagram.so")

if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(HIP):
    print("Compiling...", end=" ")
    r = subprocess.run(["hipcc", "-shared", "-fPIC", "-O2", "--offload-arch=gfx950", "-o", SO, HIP],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAIL\n{r.stderr}"); sys.exit(1)
    print("OK")

lib = ctypes.CDLL(SO)
out = torch.zeros(16 * 8, dtype=torch.float32, device="cuda")
lib.launch_test(ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(0))
torch.cuda.synchronize()
out = out.cpu().view(16, 8)

print("DS_READ_B64_TR_B16 Diagram Verification")
print("=" * 65)
print(f"LDS [4][16], val = row*16 + col\n")

# Show LDS content
for r in range(4):
    vals = [r * 16 + c for c in range(16)]
    print(f"  Row {r}: {vals}")

print(f"\nPer-lane address: lane i → &lds[i/4][(i%4)*4]\n")
print(f"  {'Lane':>4} {'row':>3} {'col0':>4}  {'ds_read_tr':>24}  {'expected':>24}  {'OK':>4}")
print(f"  {'-'*68}")

all_pass = True
for lane in range(16):
    tr = [int(v) for v in out[lane, 0:4].tolist()]
    ex = [int(v) for v in out[lane, 4:8].tolist()]
    ok = tr == ex
    if not ok: all_pass = False
    my_row = lane // 4
    my_col = (lane % 4) * 4
    print(f"  t{lane:<3d} {my_row:3d} {my_col:4d}  {str(tr):>24}  {str(ex):>24}  {'✓' if ok else '✗':>4}")

print(f"\n{'✅ ALL MATCH — diagram is correct!' if all_pass else '❌ MISMATCH'}")

if all_pass:
    print(f"\nTranspose pattern verified:")
    print(f"  t0  reads LDS cols 0-3  at row 0 → gets col 0 from rows 0,1,2,3 = {[int(v) for v in out[0, 0:4].tolist()]}")
    print(f"  t5  reads LDS cols 4-7  at row 1 → gets col 5 from rows 0,1,2,3 = {[int(v) for v in out[5, 0:4].tolist()]}")
    print(f"  t15 reads LDS cols 12-15 at row 3 → gets col 15 from rows 0,1,2,3 = {[int(v) for v in out[15, 0:4].tolist()]}")
