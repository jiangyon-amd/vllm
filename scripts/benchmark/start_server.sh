#!/bin/bash
# 快速启动 vLLM server
# Usage: bash start_server.sh <mode> [GPU_ID]
#
# Modes:
#   rtn         - RTN baseline (no rotation)
#   fused       - Fused rotation (gluon_kw8 + moe_sort)
#   hip_mfma    - HIP MFMA 3-in-1 (fastest)
#   separated   - Separated rotation (Python matmul fallback)

MODE=${1:-fused}
GPU=${2:-0}
PORT=8200

BASE_ENV="VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1 HIP_VISIBLE_DEVICES=$GPU"
COMMON="--port $PORT --trust-remote-code --disable-log-requests --max-model-len 4096 --gpu-memory-utilization 0.85 --host 0.0.0.0"

RTN_MODEL=qwen3-30b-mxfp4-rtn
ROT_MODEL=qwen3-30b-mxfp4-trained-r128-vllm-v2

case $MODE in
    rtn)
        echo "Starting RTN on GPU $GPU..."
        eval "$BASE_ENV python3 -m vllm.entrypoints.openai.api_server --model $RTN_MODEL $COMMON"
        ;;
    fused)
        echo "Starting Fused on GPU $GPU..."
        eval "VLLM_MOE_FUSED_ROTATION=1 $BASE_ENV python3 -m vllm.entrypoints.openai.api_server --model $ROT_MODEL $COMMON"
        ;;
    hip_mfma)
        echo "Starting HIP-MFMA on GPU $GPU..."
        eval "VLLM_MOE_FUSED_ROTATION=1 VLLM_MOE_HIP_MFMA=1 $BASE_ENV python3 -m vllm.entrypoints.openai.api_server --model $ROT_MODEL $COMMON"
        ;;
    separated)
        echo "Starting Separated on GPU $GPU..."
        eval "VLLM_MOE_FUSED_ROTATION=0 $BASE_ENV python3 -m vllm.entrypoints.openai.api_server --model $ROT_MODEL $COMMON"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: bash start_server.sh <rtn|fused|hip_mfma|separated> [GPU_ID]"
        exit 1
        ;;
esac
