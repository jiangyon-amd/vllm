#!/bin/bash
# 编译 HIP MFMA kernel
# Usage: bash scripts/compile_hip_kernel.sh

set -e

HIP_DIR=hip_rotation_quant
echo "=== Compiling HIP kernels ==="

echo "1. MoE MFMA 3-in-1..."
hipcc --offload-arch=gfx950 -shared -fPIC -O3 \
    -o $HIP_DIR/mfma_rot_quant_moe_sort.so \
    $HIP_DIR/mfma_rot_quant_moe_sort.hip
echo "   OK: $HIP_DIR/mfma_rot_quant_moe_sort.so"

echo "2. MoE Scalar 3-in-1..."
hipcc --offload-arch=gfx950 -shared -fPIC -O3 \
    -o $HIP_DIR/fused_rot_quant_sort_m_gt1.so \
    $HIP_DIR/fused_rot_quant_sort_m_gt1.hip
echo "   OK: $HIP_DIR/fused_rot_quant_sort_m_gt1.so"

echo "3. Dense Scalar..."
hipcc --offload-arch=gfx950 -shared -fPIC -O3 \
    -o $HIP_DIR/rotation_quant_mfma.so \
    $HIP_DIR/rotation_quant_mfma.hip 2>/dev/null && echo "   OK" || echo "   SKIP (needs torch extension)"

echo ""
echo "=== Done ==="
