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

# 4. Setup aiter HIP MFMA module
# 重要: 只复制文件，绝不 pip install aiter！否则会重编译所有模块导致 CK 版本不匹配
echo "--- 4. Setting up aiter HIP MFMA module ---"
AITER_DIST=$(python3 -c "import aiter; import os; print(os.path.dirname(aiter.__file__))" 2>/dev/null)
if [ -z "$AITER_DIST" ]; then
    echo "  WARNING: aiter not installed, skip"
else
    AITER_SRC=/data/jiangyon/aiter
    AITER_META=$(python3 -c "import aiter_meta; import os; print(os.path.dirname(aiter_meta.__file__))" 2>/dev/null)

    # a. Python wrapper
    cp $AITER_SRC/aiter/ops/mfma_rot_quant_moe_sort.py $AITER_DIST/ops/ && echo "  Python wrapper ✓"

    # b. Build config (合并新模块配置到现有 config)
    python3 -c "
import json, sys
dst = '$AITER_DIST/jit/optCompilerConfig.json'
with open(dst) as f: cfg = json.load(f)
if 'module_mfma_rot_quant_moe_sort' not in cfg:
    cfg['module_mfma_rot_quant_moe_sort'] = {
        'srcs': [
            \"f'{AITER_CSRC_DIR}/kernels/mfma_rot_quant_moe_sort.cu'\",
            \"f'{AITER_CSRC_DIR}/pybind/mfma_rot_quant_moe_sort_pybind.cu'\"
        ],
        'flags_extra_cc': [], 'flags_extra_hip': [],
        'extra_ldflags': 'None', 'extra_include': [],
        'verbose': 'False', 'blob_gen_cmd': \"''\"
    }
    with open(dst, 'w') as f: json.dump(cfg, f, indent=4)
    print('  build config ✓')
else:
    print('  build config (already exists) ✓')
"

    # c. Kernel source files (放到 aiter_meta csrc，跟其他 kernel 一致)
    if [ -n "$AITER_META" ]; then
        CSRC=$AITER_META/csrc
    else
        CSRC=$AITER_DIST/../aiter_meta/csrc
    fi
    mkdir -p $CSRC/kernels $CSRC/pybind $CSRC/include 2>/dev/null
    cp $AITER_SRC/csrc/kernels/mfma_rot_quant_moe_sort.cu $CSRC/kernels/ && echo "  kernel source ✓"
    cp $AITER_SRC/csrc/pybind/mfma_rot_quant_moe_sort_pybind.cu $CSRC/pybind/ && echo "  pybind source ✓"
    cp $AITER_SRC/csrc/include/mfma_rot_quant_moe_sort.h $CSRC/include/ && echo "  header ✓"

    # d. rocm_ops.hpp (添加 pybind 宏，如果还没有)
    if ! grep -q "MFMA_ROT_QUANT_MOE_SORT_PYBIND" $CSRC/include/rocm_ops.hpp 2>/dev/null; then
        sed -i '/#define MOE_SORTING_PYBIND/i \
#define MFMA_ROT_QUANT_MOE_SORT_PYBIND                  \\\
    m.def("mfma_rot_quant_moe_sort",                    \\\
          \&mfma_rot_quant_moe_sort,                     \\\
          py::arg("x"),                                 \\\
          py::arg("rotation"),                          \\\
          py::arg("fp4_out"),                           \\\
          py::arg("sorted_scale_out"),                  \\\
          py::arg("sorted_ids"),                        \\\
          py::arg("num_valid_ids"),                     \\\
          py::arg("token_num"),                         \\\
          py::arg("rotation_size"));\n' $CSRC/include/rocm_ops.hpp
        echo "  rocm_ops.hpp ✓"
    else
        echo "  rocm_ops.hpp (already has macro) ✓"
    fi

    # e. __init__.py import
    if ! grep -q "mfma_rot_quant_moe_sort" $AITER_DIST/__init__.py 2>/dev/null; then
        sed -i '/from .ops.moe_sorting import/a from .ops.mfma_rot_quant_moe_sort import *  # noqa: F403,E402' $AITER_DIST/__init__.py
        echo "  __init__.py import ✓"
    else
        echo "  __init__.py (already imported) ✓"
    fi

    echo "  aiter HIP MFMA module setup done (JIT will compile on first use)"
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
