#!/usr/bin/env python3
"""Test HIP MoE rot+quant+sort kernels: correctness + performance."""
import ctypes, torch, time, os

os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")

from aiter.fused_moe import moe_sorting, moe_mxfp4_sort, fused_dynamic_mxfp4_quant_moe_sort
from aiter.utility import dtypes
from vllm.model_executor.layers.quantization.quark.fused_rotation_quant_gluon_v2_kw8 import fused_gluon_v2_kw8

torch.manual_seed(42)
device = "cuda"
E, topk, K, RS, QG = 128, 8, 2048, 128, 32
n_i = K // QG

# Load HIP .so via ctypes
hip_dir = "/data/jiangyon/vllm_rotation/hip_rotation_quant"

lib_mfma = ctypes.CDLL(f"{hip_dir}/mfma_rot_quant_moe_sort.so")
lib_m_gt1 = ctypes.CDLL(f"{hip_dir}/fused_rot_quant_sort_m_gt1.so")

def get_stream():
    return torch.cuda.current_stream().cuda_stream

def launch_mfma_sort(fp4, sc, x, rot, si, nv, M, K, tn, m_o, m_pad, sx, sfp4, scm, scn, n_i):
    lib_mfma.launch_mfma_rot_quant_moe_sort(
        ctypes.c_void_p(fp4.data_ptr()),
        ctypes.c_void_p(sc.data_ptr()),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(rot.data_ptr()),
        ctypes.c_void_p(si.data_ptr()),
        ctypes.c_void_p(nv.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(tn),
        ctypes.c_int(m_o), ctypes.c_int(m_pad),
        ctypes.c_int(sx), ctypes.c_int(sfp4),
        ctypes.c_int(scm), ctypes.c_int(scn),
        ctypes.c_int(n_i),
        ctypes.c_void_p(get_stream()),
    )

def launch_m_gt1_sort(fp4, sc, x, rot, si, nv, M, K, tn, m_o, m_pad, sxm, sfp4m, scm, scn, n_i):
    lib_m_gt1.launch_fused_rot_quant_sort_m_gt1(
        ctypes.c_void_p(fp4.data_ptr()),
        ctypes.c_void_p(sc.data_ptr()),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(rot.data_ptr()),
        ctypes.c_void_p(si.data_ptr()),
        ctypes.c_void_p(nv.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(tn),
        ctypes.c_int(m_o), ctypes.c_int(m_pad),
        ctypes.c_int(sxm), ctypes.c_int(sfp4m),
        ctypes.c_int(scm), ctypes.c_int(scn),
        ctypes.c_int(n_i),
        ctypes.c_void_p(get_stream()),
    )


print("=" * 70)
print("HIP MoE rot+quant+sort kernel test")
print("=" * 70)

iters = 200
print(f"\n{'M':>4} {'gluon+sort':>12} {'hip_mfma':>12} {'hip_m_gt1':>12} {'mfma/ref':>10} {'mgt1/ref':>10}")
print("-" * 62)

for M in [1, 2, 4, 8, 16, 32]:
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    rot = torch.randn(RS, RS, dtype=torch.bfloat16, device=device) * 0.01
    topk_ids = torch.randint(0, E, (M, topk), dtype=torch.int32, device=device)
    topk_w = torch.ones(M, topk, dtype=torch.float32, device=device) / topk
    si, _, _, nv, _ = moe_sorting(topk_ids, topk_w, E, K, moebuf_dtype=x.dtype, block_size=32)

    m_o = si.shape[0]
    m_pad = ((m_o + 31) // 32) * 32

    fp4_ref = torch.empty((M, K // 2), dtype=torch.uint8, device=device)
    sc_ref = torch.empty((M, n_i), dtype=torch.uint8, device=device)
    fp4_h1 = torch.empty((M, K // 2), dtype=torch.uint8, device=device)
    sc_h1 = torch.zeros((m_pad, n_i), dtype=torch.uint8, device=device)
    fp4_h2 = torch.empty((M, K // 2), dtype=torch.uint8, device=device)
    sc_h2 = torch.zeros((m_pad, n_i), dtype=torch.uint8, device=device)

    # Warmup
    for _ in range(3):
        fused_gluon_v2_kw8(x, rot, RS, fp4_out=fp4_ref, scales_out=sc_ref, shuffle_scales=False)
        moe_mxfp4_sort(sc_ref, sorted_ids=si, num_valid_ids=nv, token_num=M, block_size=32)
        launch_mfma_sort(fp4_h1, sc_h1, x, rot, si, nv, M, K, M, m_o, m_pad,
                         x.stride(0), fp4_h1.stride(0), sc_h1.stride(0), sc_h1.stride(1), n_i)
        launch_m_gt1_sort(fp4_h2, sc_h2, x, rot, si, nv, M, K, M, m_o, m_pad,
                          x.stride(0), fp4_h2.stride(0), sc_h2.stride(0), sc_h2.stride(1), n_i)
    torch.cuda.synchronize()

    # Bench: gluon_kw8 + moe_mxfp4_sort
    t0 = time.perf_counter()
    for _ in range(iters):
        fused_gluon_v2_kw8(x, rot, RS, fp4_out=fp4_ref, scales_out=sc_ref, shuffle_scales=False)
        moe_mxfp4_sort(sc_ref, sorted_ids=si, num_valid_ids=nv, token_num=M, block_size=32)
    torch.cuda.synchronize()
    t_ref = (time.perf_counter() - t0) / iters * 1e6

    # Bench: HIP mfma sort
    t0 = time.perf_counter()
    for _ in range(iters):
        launch_mfma_sort(fp4_h1, sc_h1, x, rot, si, nv, M, K, M, m_o, m_pad,
                         x.stride(0), fp4_h1.stride(0), sc_h1.stride(0), sc_h1.stride(1), n_i)
    torch.cuda.synchronize()
    t_h1 = (time.perf_counter() - t0) / iters * 1e6

    # Bench: HIP m_gt1 sort
    t0 = time.perf_counter()
    for _ in range(iters):
        launch_m_gt1_sort(fp4_h2, sc_h2, x, rot, si, nv, M, K, M, m_o, m_pad,
                          x.stride(0), fp4_h2.stride(0), sc_h2.stride(0), sc_h2.stride(1), n_i)
    torch.cuda.synchronize()
    t_h2 = (time.perf_counter() - t0) / iters * 1e6

    print(f"M={M:>2}: {t_ref:>10.1f}µs {t_h1:>10.1f}µs {t_h2:>10.1f}µs {t_h1/t_ref:>8.2f}x {t_h2/t_ref:>8.2f}x")

# Correctness check for M=4
print("\n--- Correctness (M=4) ---")
M = 4
x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
rot = torch.randn(RS, RS, dtype=torch.bfloat16, device=device) * 0.01
topk_ids = torch.randint(0, E, (M, topk), dtype=torch.int32, device=device)
topk_w = torch.ones(M, topk, dtype=torch.float32, device=device) / topk
si, _, _, nv, _ = moe_sorting(topk_ids, topk_w, E, K, moebuf_dtype=x.dtype, block_size=32)
m_o = si.shape[0]; m_pad = ((m_o + 31) // 32) * 32

# Reference: Python rot + aiter quant+sort
x_rot = (x.reshape(-1, K//RS, RS) @ rot).reshape(M, K)
fp4_aiter, sc_aiter = fused_dynamic_mxfp4_quant_moe_sort(
    x_rot, sorted_ids=si, num_valid_ids=nv, token_num=M, topk=1, block_size=32)

# HIP mfma
fp4_h1 = torch.empty((M, K // 2), dtype=torch.uint8, device=device)
sc_h1 = torch.zeros((m_pad, n_i), dtype=torch.uint8, device=device)
launch_mfma_sort(fp4_h1, sc_h1, x, rot, si, nv, M, K, M, m_o, m_pad,
                 x.stride(0), fp4_h1.stride(0), sc_h1.stride(0), sc_h1.stride(1), n_i)

# HIP m_gt1
fp4_h2 = torch.empty((M, K // 2), dtype=torch.uint8, device=device)
sc_h2 = torch.zeros((m_pad, n_i), dtype=torch.uint8, device=device)
launch_m_gt1_sort(fp4_h2, sc_h2, x, rot, si, nv, M, K, M, m_o, m_pad,
                  x.stride(0), fp4_h2.stride(0), sc_h2.stride(0), sc_h2.stride(1), n_i)
torch.cuda.synchronize()

sc_a = sc_aiter.view(torch.uint8)[:m_pad, :n_i]
sc_1 = sc_h1[:m_pad, :n_i]
sc_2 = sc_h2[:m_pad, :n_i]

m1 = (sc_a == sc_1).sum().item() / sc_a.numel() * 100
m2 = (sc_a == sc_2).sum().item() / sc_a.numel() * 100
print(f"  mfma  scale match: {m1:.1f}%")
print(f"  m_gt1 scale match: {m2:.1f}%")
