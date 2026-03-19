#!/bin/bash
# 一键 E2E Benchmark: RTN / Fused / HIP-MFMA
# Usage: bash run_all.sh [GPU_ID]
#
# 前提: vLLM 和 aiter 已安装，模型已量化

GPU=${1:-1}
echo "Using GPU $GPU"
echo ""

# 1. 检查 GPU
echo "=== Step 1: Check GPU ==="
python3 scripts/benchmark/check_gpu.py 2>/dev/null || \
    HIP_VISIBLE_DEVICES=$GPU python3 -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'GPU {0}: {f/1e9:.1f}GB free / {t/1e9:.1f}GB total')"
echo ""

# 2. Kernel 级 Benchmark (不需要 server)
echo "=== Step 2: Kernel Benchmark ==="
echo "--- HIP MFMA kernel ---"
HIP_VISIBLE_DEVICES=$GPU python3 scripts/benchmark/test_hip_mfma_3in1.py 2>&1 | grep -E "M=|Overall|MFMA"
echo ""

# 3. E2E Benchmark (需要 server)
echo "=== Step 3: E2E Benchmark ==="
echo "Running: RTN → Fused → HIP-MFMA"
python3 scripts/benchmark/run_benchmark.py --gpu $GPU --modes rtn fused hip_mfma
echo ""

echo "=== Done ==="
echo "Results in: bench_results/"
