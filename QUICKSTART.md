# Fused Rotation Quantization - Quick Start Guide

在新 container 中快速复现 MoE/Dense rotation quantization 的完整流程。

## 1. 环境要求

```
- GPU: AMD Instinct MI355X (gfx950) 或 MI300X (gfx942)
- ROCm: 7.0+
- Python: 3.12
- PyTorch: 2.9+ (ROCm build)
- Triton: 3.5.x (3.6 有 gl.convert_layout bug)
```

## 2. 安装

```bash
# vLLM (editable install)
cd /data/jiangyon/vllm_rotation
VLLM_TARGET_DEVICE=empty SETUPTOOLS_SCM_PRETEND_VERSION=0.14.0 pip install -e . --no-build-isolation

# AMD Quark (量化工具)
cd /data/jiangyon/vllm_rotation/quark_examples
pip install -e .

# aiter (AMD GPU kernel library) - 使用预编译版本，不要从源码 pip install
# 如需添加 HIP MFMA kernel，只复制文件，不要重新编译整个 aiter
```

## 3. 模型路径

### BF16 源模型
| Model | Path |
|-------|------|
| Qwen3-8B | `/data/jiangyon/hf_cache/Qwen3-8B` |
| Qwen3-32B | `/data/jiangyon/hf_cache/Qwen3-32B` |
| Qwen3-30B-A3B (MoE) | `/data/jiangyon/hf_cache/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |

### MXFP4 量化模型
| Model | Path | 说明 |
|-------|------|------|
| 8B RTN | `qwen3-8b-mxfp4-rtn` | 无 rotation，基线 |
| 8B Trained | `qwen3-8b-mxfp4-trained-r128-vllm` | 训练 rotation |
| 32B RTN | `qwen3-32b-mxfp4-rtn` | 无 rotation，基线 |
| 30B MoE RTN | `qwen3-30b-mxfp4-rtn` | 无 rotation，MoE 基线 |
| 30B MoE Trained | `qwen3-30b-mxfp4-trained-r128-vllm-v2` | 训练 rotation，**推荐** |

所有量化模型路径相对于 `/data/jiangyon/vllm_rotation/`。

## 4. 量化脚本

### 4.1 RTN 量化 (无 rotation，基线)
```bash
cd quark_examples/examples/torch/language_modeling/llm_ptq
HIP_VISIBLE_DEVICES=0 python quantize_quark.py \
    --model_dir <BF16_MODEL_PATH> --quant_scheme mxfp4 \
    --model_export hf_format --output_dir <OUTPUT_DIR> \
    --model_attn_implementation sdpa --skip_evaluation
```

### 4.2 训练 Rotation + 量化 (两步)
```bash
cd quark_examples/examples/torch/language_modeling/rotation

# Step 1: 训练 rotation matrices
# Dense 8B:
HIP_VISIBLE_DEVICES=0 python train_rotation.py \
    --model_dir /data/jiangyon/hf_cache/Qwen3-8B --quant_scheme mxfp4 \
    --rotation_algo_config_file ./qwen3_train_r1_128_online_r2.json \
    --export_rotation --max_steps 10 --loss_type kl_top_1000 \
    --learning_rate 1.5 --num_samples 4000 --train_batch_size 8 \
    --eval_batch_size 16 --model_attn_implementation sdpa \
    --output_dir /tmp/rotation_output

# MoE 30B (加 --force_custom_architecture):
HIP_VISIBLE_DEVICES=0 python train_rotation.py \
    --model_dir /data/jiangyon/hf_cache/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 \
    --quant_scheme mxfp4 \
    --rotation_algo_config_file ./qwen3_moe_30b_train_r1_128_online_r2.json \
    --export_rotation --max_steps 10 --loss_type kl_top_1000 \
    --learning_rate 1.5 --num_samples 4000 --train_batch_size 8 \
    --eval_batch_size 16 --model_attn_implementation sdpa \
    --force_custom_architecture --output_dir /tmp/rotation_output

# Step 2: 导出带 rotation 的量化模型
HIP_VISIBLE_DEVICES=0 python train_rotation.py \
    --model_dir <BF16_MODEL> --quant_scheme mxfp4 \
    --rotation_algo_config_file <CONFIG>.json \
    --pretrained_rotation_path /tmp/rotation_output/rotations.safetensors \
    --skip_training --export_rotation --model_export hf_format \
    --model_attn_implementation sdpa --force_custom_architecture \
    --output_dir <FINAL_OUTPUT_DIR>
```

### Rotation config 文件
- Dense: `quark_examples/examples/torch/language_modeling/rotation/qwen3_train_r1_128_online_r2.json`
- MoE: `quark_examples/examples/torch/language_modeling/rotation/qwen3_moe_30b_train_r1_128_online_r2.json`

## 5. vLLM Server 启动

### 通用参数
```bash
BASE_ENV="VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1"
COMMON_ARGS="--port 8200 --trust-remote-code --disable-log-requests --max-model-len 4096 --gpu-memory-utilization 0.85 --host 0.0.0.0"
```

### Dense 模型
```bash
# RTN (无 rotation)
$BASE_ENV HIP_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model qwen3-8b-mxfp4-rtn $COMMON_ARGS

# Fused rotation (默认 gluon v2 kernel)
$BASE_ENV HIP_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model qwen3-8b-mxfp4-trained-r128-vllm $COMMON_ARGS

# Separated rotation (Python matmul fallback)
VLLM_DISABLE_FUSED_ROT_QUANT=1 $BASE_ENV HIP_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model qwen3-8b-mxfp4-trained-r128-vllm $COMMON_ARGS
```

### MoE 模型
```bash
MODEL=qwen3-30b-mxfp4-trained-r128-vllm-v2

# RTN 基线
$BASE_ENV HIP_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model qwen3-30b-mxfp4-rtn $COMMON_ARGS

# Fused (gluon_kw8 + moe_mxfp4_sort, 默认)
VLLM_MOE_FUSED_ROTATION=1 $BASE_ENV HIP_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL $COMMON_ARGS

# HIP MFMA 3-in-1 (最快，需要 aiter 集成)
VLLM_MOE_FUSED_ROTATION=1 VLLM_MOE_HIP_MFMA=1 $BASE_ENV HIP_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL $COMMON_ARGS

# Separated (Python rotation matmul fallback)
VLLM_MOE_FUSED_ROTATION=0 $BASE_ENV HIP_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL $COMMON_ARGS
```

## 6. Benchmark

### 正确性检查
```bash
curl -s http://localhost:8200/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"MODEL_PATH","messages":[{"role":"user","content":"只回答数字：1+1等于几？"}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

### E2E Benchmark (使用 skill 脚本)
```bash
python3 /data/jiangyon/.cursor/skills/gpu-benchmark-moe/scripts/run_benchmark.py \
    --gpu 0 --modes rtn fused hip_mfma
```

### Kernel 级 Benchmark
```bash
HIP_VISIBLE_DEVICES=0 python3 /data/jiangyon/GEAK/test_hip_mfma_3in1.py
```

## 7. HIP MFMA Kernel 集成到 aiter

**重要**: 不要用 `pip install .` 从 aiter 源码安装！会破坏预编译的 JIT 模块。

只需复制以下文件到已安装的 aiter 中:

```bash
AITER_DIST=/usr/local/lib/python3.12/dist-packages/aiter
AITER_SRC=/data/jiangyon/aiter

# 1. Kernel 源码
cp $AITER_SRC/csrc/kernels/mfma_rot_quant_moe_sort.cu $(dirname $AITER_DIST)/../aiter_meta/csrc/kernels/ 2>/dev/null || \
    mkdir -p $AITER_DIST/csrc/kernels && cp $AITER_SRC/csrc/kernels/mfma_rot_quant_moe_sort.cu $AITER_DIST/csrc/kernels/

# 2. Header
cp $AITER_SRC/csrc/include/mfma_rot_quant_moe_sort.h $(dirname $AITER_DIST)/../aiter_meta/csrc/include/ 2>/dev/null || \
    mkdir -p $AITER_DIST/csrc/include && cp $AITER_SRC/csrc/include/mfma_rot_quant_moe_sort.h $AITER_DIST/csrc/include/

# 3. Pybind
cp $AITER_SRC/csrc/pybind/mfma_rot_quant_moe_sort_pybind.cu $(dirname $AITER_DIST)/../aiter_meta/csrc/pybind/ 2>/dev/null || \
    mkdir -p $AITER_DIST/csrc/pybind && cp $AITER_SRC/csrc/pybind/mfma_rot_quant_moe_sort_pybind.cu $AITER_DIST/csrc/pybind/

# 4. Python wrapper
cp $AITER_SRC/aiter/ops/mfma_rot_quant_moe_sort.py $AITER_DIST/ops/

# 5. Build config
cp $AITER_SRC/aiter/jit/optCompilerConfig.json $AITER_DIST/jit/

# 6. 添加 import (如果还没有)
grep -q "mfma_rot_quant_moe_sort" $AITER_DIST/__init__.py || \
    sed -i '/from .ops.moe_sorting import/a from .ops.mfma_rot_quant_moe_sort import *  # noqa: F403,E402' $AITER_DIST/__init__.py

# 7. 更新 rocm_ops.hpp (添加 MFMA_ROT_QUANT_MOE_SORT_PYBIND 宏)
# 参考: /data/jiangyon/aiter/csrc/include/rocm_ops.hpp
```

## 8. 环境变量一览

| 变量 | 默认 | 说明 |
|------|------|------|
| `VLLM_ROCM_USE_AITER` | 0 | 启用 aiter GPU kernels |
| `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM` | 0 | 启用 FP4 ASM GEMM |
| `VLLM_MOE_FUSED_ROTATION` | 1 | MoE fused rotation (默认开) |
| `VLLM_MOE_HIP_MFMA` | 0 | 启用 HIP MFMA 3-in-1 kernel |
| `VLLM_MOE_GLUON_DECODE` | 1 | M=1 decode 用 Gluon kernel |
| `VLLM_MOE_GLUON_KW8` | 1 | M>1 用 gluon_kw8 + moe_mxfp4_sort |
| `VLLM_DISABLE_FUSED_ROT_QUANT` | 0 | Dense: 禁用 fused 走 separated |
| `HIP_VISIBLE_DEVICES` | - | 指定 GPU |

## 9. 关键文件

### vllm_rotation 代码
```
vllm/model_executor/layers/quantization/quark/
├── fused_rotation_mxfp4_quant_moe_sort.py    # MoE dispatch (核心)
├── fused_rotation_quant_mfma_hip.py           # HIP MFMA ctypes wrapper (备用)
├── fused_rotation_quant_gluon_v2_kw8.py       # Gluon v2 kw8 kernel (M>1)
├── fused_rotation_mxfp4_quant_moe_gluon.py    # Gluon MoE M=1 kernel
├── fused_rotation_quant_sort_triton.py         # Triton 3-in-1 (参考)
├── quark_moe.py                                # MoE pipeline integration
└── transform.py                                # Quark format handling
```

### HIP Kernel
```
hip_rotation_quant/
├── mfma_rot_quant_moe_sort.hip    # MFMA MoE 3-in-1 (最快, 5-13µs)
├── fused_rot_quant_sort_m_gt1.hip # Scalar MoE 3-in-1 (100% 精确)
└── rotation_quant_mfma.hip        # Dense scalar kernel
```

### aiter 集成
```
aiter/
├── csrc/kernels/mfma_rot_quant_moe_sort.cu
├── csrc/include/mfma_rot_quant_moe_sort.h
├── csrc/pybind/mfma_rot_quant_moe_sort_pybind.cu
└── aiter/ops/mfma_rot_quant_moe_sort.py
```

### GEAK (AI kernel optimizer)
```
/data/jiangyon/GEAK/
├── test_hip_mfma_3in1.py          # MFMA kernel 测试 (vs torch.matmul)
├── test_hip_moe_3in1.py           # Scalar kernel 测试 (vs gluon ref)
├── task_moe_mfma_3in1.txt         # GEAK 任务描述
└── knowledge-base/amd-knowledge-base/layer-2-compute-stack/hip/
    └── mfma-bf16-16x16x32-layout.md  # MFMA + ds_read_tr 文档
```

## 10. Kernel 性能 (MI355X)

### MoE HIP MFMA 3-in-1 (via aiter torch op)
| M | 2-kernel 基线 | HIP MFMA | 加速 |
|---|-------------|----------|------|
| 1 | 28.8µs | **5.5µs** | 5.3x |
| 8 | 28.7µs | **9.6µs** | 3.0x |
| 32 | 28.6µs | **13.1µs** | 2.2x |
| 256 | 28.8µs | **12.1µs** | 2.4x |

### Dense Gluon v2 (fused shuffle)
| Kernel | Time |
|--------|------|
| Gluon v2 fused | 14.5µs |
| Separated | 21µs |

## 11. MFMA 技术要点

### v_mfma_f32_16x16x32_bf16 操作数布局
```
A: lane%16 = M-row,  lane/16 = k_group,  8 bf16/lane
B: lane%16 = N-col,  lane/16 = k_group,  8 bf16/lane
C: lane%16 = col,    row = (lane/16)*4+i, i=0..3
```

### ds_read_b64_tr_b16 用法
```
Per-lane 地址: lane i → &block[i/4][(i%4)*4]
LDS 必须 pre-tile: R_tiled[N_TILES][K][16] (行步长 = 32 bytes)
两次 ds_read_tr → 8 bf16/lane = MFMA B 操作数
```
