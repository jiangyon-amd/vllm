#!/bin/bash
# RTN 量化 (无 rotation，基线)
# Usage: bash scripts/quantize/quantize_rtn.sh <BF16_MODEL> <OUTPUT_DIR> [GPU]
#
# Example:
#   bash scripts/quantize/quantize_rtn.sh /data/jiangyon/hf_cache/Qwen3-8B qwen3-8b-mxfp4-rtn 0
#   bash scripts/quantize/quantize_rtn.sh /data/jiangyon/hf_cache/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 qwen3-30b-mxfp4-rtn 0

set -e

MODEL=${1:?Usage: $0 <BF16_MODEL> <OUTPUT_DIR> [GPU]}
OUTPUT=${2:?Usage: $0 <BF16_MODEL> <OUTPUT_DIR> [GPU]}
GPU=${3:-0}

echo "=== RTN Quantization ==="
echo "Model: $MODEL"
echo "Output: $OUTPUT"
echo "GPU: $GPU"

cd quark_examples/examples/torch/language_modeling/llm_ptq

HIP_VISIBLE_DEVICES=$GPU python quantize_quark.py \
    --model_dir $MODEL \
    --quant_scheme mxfp4 \
    --model_export hf_format \
    --output_dir $OUTPUT \
    --model_attn_implementation sdpa \
    --skip_evaluation

echo "=== Done: $OUTPUT ==="
