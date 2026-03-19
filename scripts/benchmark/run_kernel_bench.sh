#!/bin/bash
# Kernel 级性能测试 (不需要启动 server)
# Usage: bash run_kernel_bench.sh [GPU_ID]

GPU=${1:-0}

echo "=== MoE HIP MFMA Kernel Benchmark (GPU $GPU) ==="
HIP_VISIBLE_DEVICES=$GPU python3 -c "
import torch, time, math, sys
sys.path.insert(0, '.')
from aiter.fused_moe import moe_sorting
from aiter.utility import dtypes
from aiter.utility.fp4_utils import moe_mxfp4_sort
from aiter.ops.mfma_rot_quant_moe_sort import mfma_rot_quant_moe_sort
from vllm.model_executor.layers.quantization.quark.fused_rotation_quant_gluon_v2_kw8 import fused_gluon_v2_kw8

E, topk, K, RS, QG = 128, 8, 2048, 128, 32
n_i, device, N = K // QG, 'cuda', 2000

print(f'{'M':>5} {'gluon+sort':>12} {'HIP MFMA':>12} {'speedup':>8}')
print('-' * 45)

for M in [1, 4, 8, 16, 32, 64, 128, 256]:
    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    rot = torch.randn(RS, RS, dtype=torch.bfloat16, device=device) * 0.01
    topk_ids = torch.randint(0, E, (M, topk), dtype=torch.int32, device=device)
    topk_w = torch.ones(M, topk, dtype=torch.float32, device=device) / topk
    si, _, _, nv, _ = moe_sorting(topk_ids, topk_w, E, K, torch.bfloat16, 32, None)
    m_o = si.shape[0]; m_pad = ((m_o+31)//32)*32

    fp4_ref = torch.empty((M, K//2), dtype=torch.uint8, device=device)
    sc_ref = torch.empty((M, n_i), dtype=torch.uint8, device=device)
    fp4_h = torch.empty((M, K//2), dtype=torch.uint8, device=device)
    sc_h = torch.zeros((m_pad, n_i), dtype=torch.uint8, device=device)

    for _ in range(200):
        fused_gluon_v2_kw8(x, rot, RS, fp4_out=fp4_ref, scales_out=sc_ref, shuffle_scales=False)
        moe_mxfp4_sort(sc_ref, sorted_ids=si, num_valid_ids=nv, token_num=M, block_size=32)
    for _ in range(200):
        mfma_rot_quant_moe_sort(x, rot, fp4_h, sc_h, si, nv, M, RS)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N):
        fused_gluon_v2_kw8(x, rot, RS, fp4_out=fp4_ref, scales_out=sc_ref, shuffle_scales=False)
        moe_mxfp4_sort(sc_ref, sorted_ids=si, num_valid_ids=nv, token_num=M, block_size=32)
    torch.cuda.synchronize()
    ref_us = (time.perf_counter() - t0) / N * 1e6

    t0 = time.perf_counter()
    for _ in range(N):
        mfma_rot_quant_moe_sort(x, rot, fp4_h, sc_h, si, nv, M, RS)
    torch.cuda.synchronize()
    hip_us = (time.perf_counter() - t0) / N * 1e6

    print(f'{M:5d} {ref_us:10.1f}us {hip_us:10.1f}us {ref_us/hip_us:7.2f}x')
"
