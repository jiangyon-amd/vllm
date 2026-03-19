#!/bin/bash
# 新 container 环境初始化
# Usage: bash scripts/setup_env.sh
#
# 前提: ROCm 7.0+, Python 3.12, PyTorch 2.9+ (ROCm build) 已安装

set -e

echo "=== Environment Setup ==="

# 1. Install vLLM (editable)
echo "--- 1. Installing vLLM ---"
cd /data/jiangyon/vllm_rotation
VLLM_TARGET_DEVICE=empty SETUPTOOLS_SCM_PRETEND_VERSION=0.14.0 pip install -e . --no-build-isolation
echo "vLLM installed"

# 2. Install Quark
echo "--- 2. Installing Quark ---"
cd /data/jiangyon/vllm_rotation/quark_examples
pip install -e .
echo "Quark installed"

# 3. Compile HIP kernels
echo "--- 3. Compiling HIP kernels ---"
cd /data/jiangyon/vllm_rotation
bash scripts/compile_hip_kernel.sh

# 4. Setup aiter HIP MFMA module (copy files only, do NOT pip install aiter from source)
echo "--- 4. Setting up aiter HIP MFMA module ---"
AITER_DIST=$(python3 -c "import aiter; import os; print(os.path.dirname(aiter.__file__))")
if [ -n "$AITER_DIST" ]; then
    AITER_SRC=/data/jiangyon/aiter

    # Copy module files
    cp $AITER_SRC/aiter/ops/mfma_rot_quant_moe_sort.py $AITER_DIST/ops/ 2>/dev/null && echo "  ops wrapper ✓"
    cp $AITER_SRC/aiter/jit/optCompilerConfig.json $AITER_DIST/jit/ 2>/dev/null && echo "  build config ✓"

    # Add import if missing
    if ! grep -q "mfma_rot_quant_moe_sort" $AITER_DIST/__init__.py 2>/dev/null; then
        sed -i '/from .ops.moe_sorting import/a from .ops.mfma_rot_quant_moe_sort import *  # noqa: F403,E402' $AITER_DIST/__init__.py
        echo "  __init__.py import ✓"
    fi

    echo "  aiter module setup done"
else
    echo "  WARNING: aiter not installed, skip HIP MFMA module setup"
fi

# 5. Verify
echo ""
echo "--- Verification ---"
python3 -c "
import vllm; print(f'vLLM: {vllm.__version__}')
import aiter; print(f'aiter: OK')
import triton; print(f'Triton: {triton.__version__}')
try:
    from aiter.ops.mfma_rot_quant_moe_sort import mfma_rot_quant_moe_sort
    print('HIP MFMA module: OK')
except: print('HIP MFMA module: NOT AVAILABLE (will JIT compile on first use)')
"

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. Kernel test:  bash scripts/benchmark/run_kernel_bench.sh 0"
echo "  2. Start server: bash scripts/benchmark/start_server.sh hip_mfma 0"
echo "  3. Full bench:   bash scripts/benchmark/run_all.sh 0"
