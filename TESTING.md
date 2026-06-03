# Testing Guide: Online R1 Rotation + MXFP4 Quantization

This document describes how to verify correctness and performance of the
fused rotation + MXFP4 quantization feature for AMD MI355X (gfx950).

## Hardware / Software Requirements

- AMD MI355X GPU (gfx950)
- ROCm with Triton 3.5.x (Gluon cdna4 support)
- aiter library with `fused_moe_2stages` API
- vllm ≥ 0.14.0

## Environment Variables

| Variable | Value | Purpose |
|---|---|---|
| `VLLM_FUSED_ROTATION` | `1` | Enable both Dense Gluon + MoE Gluon kernels (**recommended**) |
| `VLLM_ROCM_USE_AITER` | `1` | Enable aiter ROCm kernels |
| `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM` | `1` | Enable preshuffled ASM GEMM path (required for fused rotation path) |

> **Note**: `VLLM_FUSED_ROTATION=1` supersedes the legacy `VLLM_USE_FUSED_ROTATION_QUANT=1` and
> `VLLM_MOE_FUSED_ROTATION=1` env vars (which were renamed/removed in this cleanup).

## Supported Model Format

The model must be quantized with online rotation enabled in the Quark config. Example:

```json
{
  "online_config": {
    "online_rotation_layers": ["model.layers.*.self_attn.q_proj", "model.layers.*.mlp.gate_up_proj"]
  }
}
```

Legacy Quark ≤0.11 format (`online_r1_rotation: true` + `scaling_layers`) is also supported.

---

## 1. Server Startup

```bash
TMPDIR=/tmp TORCH_EXTENSIONS_DIR=/tmp/torch_ext \
TRITON_CACHE_DIR=/tmp/triton_cache VLLM_CACHE_ROOT=/tmp/vllm_cache \
HF_HOME=/workspace/hf_cache VLLM_NO_USAGE_STATS=1 \
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1 \
VLLM_FUSED_ROTATION=1 \
python3 -m vllm.entrypoints.openai.api_server \
    --model /path/to/qwen3-30b-mxfp4-hadamard-vllm \
    --served-model-name qwen3 \
    --port 18310 --host 0.0.0.0 \
    --trust-remote-code --disable-log-requests \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.87
```

**Expected startup log messages** (confirms both kernels active):
```
INFO [quark_ocp_mx.py] Using fused Triton rotation+MXFP4 quant kernel
INFO [fused_moe_rotation.py] Applied Gluon rotation patch to aiter.fused_moe_2stages
INFO [qwen3_moe.py] MoE rotation enabled for model.layers.0.mlp (rotation_size=128, mode=fused)
```

---

## 2. Correctness Check

### 2a. Basic inference

```bash
curl -s http://localhost:18310/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3",
        "messages": [{"role": "user", "content": "1+1等于几？只回答数字。"}],
        "max_tokens": 32, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": false}
    }' | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
# Expected: 2
```

### 2b. Perplexity (PPL) evaluation

Uses WikiText-2 test set, seqlen=2048, 146 chunks.

```bash
python3 vllm_ppl_quark_style.py \
    --model /path/to/qwen3-30b-mxfp4-hadamard-vllm \
    --url http://localhost:18310
```

**Expected results** (Qwen3-30B-A3B Hadamard MXFP4):

| Mode | PPL |
|---|---|
| Fused (`VLLM_FUSED_ROTATION=1`) | 9.77 |
| Separated (no fused rotation) | 9.79 |
| Offline Quark baseline | 9.72 |

> PPL difference fused vs separated < 0.05 is expected (minor floating-point ordering difference).

---

## 3. Throughput Benchmark

Matching the final_fixed baseline parameters:
- Dataset: random, ISL=1024, OSL=1024
- 10 warmup requests, 20 measurement requests per concurrency

```bash
MODEL_PATH=/path/to/qwen3-30b-mxfp4-hadamard-vllm

for C in 1 2 4 8 16 32; do
    echo -n "c=$C: "
    vllm bench serve \
        --backend openai \
        --base-url http://localhost:18310 \
        --model qwen3 \
        --tokenizer $MODEL_PATH \
        --dataset-name random \
        --random-input-len 1024 \
        --random-output-len 1024 \
        --num-prompts 20 \
        --num-warmups 10 \
        --max-concurrency $C \
        --trust-remote-code 2>&1 | grep -E "Total token throughput|Median TPOT"
done
```

**Expected results** (Qwen3-30B-A3B Hadamard MXFP4, AMD MI355X):

| Concurrency | tok/s | Median TPOT (ms) |
|---|---|---|
| 1  | ~288 | ~6.9 |
| 2  | ~497 | ~8.0 |
| 4  | ~956 | ~8.4 |
| 8  | ~1508 | ~8.9 |
| 16 | ~2222 | ~9.5 |
| 32 | ~3847 | ~10.3 |

> These numbers reflect cleanup branch (2026-06-02) with LDS round-trip scale shuffle optimization.
> Baseline (final_fixed, before LDS shuffle): c=1: 239.6 tok/s / 8.35ms, c=32: 3515.6 tok/s / 9.86ms.
> Cleanup branch is ≥ baseline (5–20% improvement at low concurrency from the LDS scale shuffle kernel).

---

## 4. Smoke Test Script

A single script to run both fused and separated PPL:

```bash
./smoke_30b_cleanup.sh
```

Expected output in `smoke_30b_cleanup_<timestamp>/summary.txt`:
```
fused PPL=9.77
separated PPL=9.79
```

---

## 5. Key Files

| File | Purpose |
|---|---|
| `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py` | Main quantization scheme with rotation dispatch |
| `vllm/model_executor/layers/quantization/quark/fused_rotation_quant_gluon.py` | Dense Gluon kernel (rotation + MXFP4 quant) |
| `vllm/model_executor/layers/quantization/quark/gluon_rotation_moe.py` | MoE rotation-only Gluon kernel |
| `vllm/model_executor/layers/quantization/quark/fused_moe_rotation.py` | MoE feature flags and monkey-patch |
| `vllm/model_executor/layers/quantization/quark/transform.py` | `get_online_rotation_layers()` helper |
| `vllm/model_executor/models/qwen3_moe.py` | Qwen3 MoE model with rotation support |

---

## 6. Benchmark 方法论说明

### ⚠️ 两种 benchmark 模式不可混用

| 模式 | 参数 | 含义 | 适用场景 |
|---|---|---|---|
| 并发模式 | `--max-concurrency N` | 最多 N 个请求同时在飞 | 测峰值吞吐、解码延迟 |
| 速率模式 | `--request-rate N` | 每秒发 N 个新请求 | 测特定负载下的稳态延迟 |

**同一 N 值下两种模式的 TPOT 数字不可直接对比**（含义不同，数量级可能相差 2-3x）。

### 统一标准（本项目后续）

使用 `--max-concurrency`，配合足够的样本量：

```bash
vllm bench serve \
    --backend openai \
    --base-url http://localhost:PORT \
    --model MODEL_PATH \
    --tokenizer MODEL_PATH \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --num-prompts 200 \      # ≥200，避免 20-prompt 噪声
    --num-warmups 20 \
    --max-concurrency 16     # 或 32
```

### 历史基线说明

`bench_32b_3way_run.log` / `bench_8b_3way_run.log` / `bench_14b_3way_run.log` 等历史日志
使用的是 `--request-rate N`（旧方式），**与上述标准不可混用**，不应作为 regression 对比基准。

可信对比基准：`bench_results/4model_sanity_*/`（2026-06-03，200 prompts, max-concurrency）。
