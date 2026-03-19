#!/bin/bash
# Dense 模型: 训练 rotation + 导出量化模型
# Usage: bash scripts/quantize/train_rotation_dense.sh <BF16_MODEL> <OUTPUT_DIR> [GPU]
#
# Example:
#   bash scripts/quantize/train_rotation_dense.sh /data/jiangyon/hf_cache/Qwen3-8B qwen3-8b-mxfp4-trained-r128 0

set -e

MODEL=${1:?Usage: $0 <BF16_MODEL> <OUTPUT_DIR> [GPU]}
OUTPUT=${2:?Usage: $0 <BF16_MODEL> <OUTPUT_DIR> [GPU]}
GPU=${3:-0}
ROTATION_DIR=${OUTPUT}-rotations

echo "=== Dense Rotation Training ==="
echo "Model: $MODEL"
echo "Output: $OUTPUT"
echo "GPU: $GPU"

cd quark_examples/examples/torch/language_modeling/rotation

# Step 1: Train rotation
echo "--- Step 1: Training rotation matrices ---"
HIP_VISIBLE_DEVICES=$GPU python train_rotation.py \
    --model_dir $MODEL \
    --quant_scheme mxfp4 \
    --rotation_algo_config_file ./qwen3_train_r1_128_online_r2.json \
    --export_rotation \
    --max_steps 10 \
    --loss_type kl_top_1000 \
    --learning_rate 1.5 \
    --num_samples 4000 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --model_attn_implementation sdpa \
    --output_dir $ROTATION_DIR

# Step 2: Export quantized model
echo "--- Step 2: Exporting quantized model ---"
HIP_VISIBLE_DEVICES=$GPU python train_rotation.py \
    --model_dir $MODEL \
    --quant_scheme mxfp4 \
    --rotation_algo_config_file ./qwen3_train_r1_128_online_r2.json \
    --pretrained_rotation_path $ROTATION_DIR/rotations.safetensors \
    --skip_training \
    --export_rotation \
    --model_export hf_format \
    --model_attn_implementation sdpa \
    --output_dir $OUTPUT

echo "=== Done: $OUTPUT ==="
