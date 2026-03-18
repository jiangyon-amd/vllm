// Standalone MFMA 16x16x32 BF16 test with self-verification
// Computes C = A @ B using MFMA, verifies against scalar reference
//
// MFMA 16x16x32 BF16 layout (from AMD matrix_calculator):
//   A operand: lane_id%16 = M-row, lane_id/16 = k_group (0..3), 8 bf16 each
//   B operand: lane_id%16 = N-col, lane_id/16 = k_group (0..3), 8 bf16 each
//   C output:  lane_id%16 = col, (lane_id/16)*4+i = row, float4 per lane
//
// Build: hipcc -shared -fPIC -O2 --offload-arch=gfx950 -o test_standalone_mfma.so test_standalone_mfma.hip
// Test:  python3 test_standalone_mfma.py

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

using float4_t = __attribute__((ext_vector_type(4))) float;
using bf16x8_t = __attribute__((ext_vector_type(8))) __bf16;

// Single-wave MFMA: 16x16 output from 16xK @ Kx16 matmul
// Uses 64 threads (1 wavefront)
__global__ __attribute__((amdgpu_flat_work_group_size(64, 64)))
void mfma_16x16_kernel(
    const __bf16* __restrict__ A,   // [M, K] row-major
    const __bf16* __restrict__ B,   // [K, N] row-major
    float* __restrict__ C,          // [M, N] row-major output
    int M, int N, int K)
{
    int lane_id = threadIdx.x;  // 0..63
    int k_group = lane_id / 16; // 0..3: which 8 of 32 K-elements
    int lane_16 = lane_id % 16; // 0..15

    float4_t acc = {};

    // Loop over K in chunks of 32 (MFMA K-dim)
    for (int k_base = 0; k_base < K; k_base += 32) {
        int a_k_start = k_base + k_group * 8;
        int a_row = lane_16;  // each lane handles one M-row

        // Load A: 8 consecutive bf16 from row a_row, starting at column a_k_start
        bf16x8_t a_data = {};
        if (a_row < M) {
            for (int j = 0; j < 8; j++) {
                int col = a_k_start + j;
                a_data[j] = (col < K) ? A[a_row * K + col] : (__bf16)__float2bfloat16(0.0f);
            }
        }

        // Load B: 8 consecutive bf16 from column lane_16, rows k_start..k_start+7
        int b_col = lane_16;
        int b_k_start = k_base + k_group * 8;
        bf16x8_t b_data = {};
        if (b_col < N) {
            for (int j = 0; j < 8; j++) {
                int row = b_k_start + j;
                b_data[j] = (row < K) ? B[row * N + b_col] : (__bf16)__float2bfloat16(0.0f);
            }
        }

        // MFMA: C += A @ B
        acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a_data, b_data, acc, 0, 0, 0);
    }

    // Store output using MFMA layout:
    //   c_col = lane_id % 16
    //   c_row = (lane_id / 16) * 4 + i   (i = 0..3 from float4)
    int c_col = lane_16;
    int c_m_group = lane_id / 16;

    for (int i = 0; i < 4; i++) {
        int c_row = c_m_group * 4 + i;
        if (c_row < M && c_col < N) {
            C[c_row * N + c_col] = acc[i];
        }
    }
}

// Scalar reference: simple matmul for verification
__global__ void scalar_matmul_kernel(
    const __bf16* __restrict__ A,
    const __bf16* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    for (int col = 0; col < N; col++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __bfloat162float(A[row * K + k]) * __bfloat162float(B[k * N + col]);
        }
        C[row * N + col] = sum;
    }
}

extern "C" {

void launch_mfma_16x16(
    float* C, const void* A, const void* B,
    int M, int N, int K, hipStream_t stream)
{
    mfma_16x16_kernel<<<1, 64, 0, stream>>>(
        (const __bf16*)A, (const __bf16*)B, C, M, N, K);
}

void launch_scalar_matmul(
    float* C, const void* A, const void* B,
    int M, int N, int K, hipStream_t stream)
{
    scalar_matmul_kernel<<<1, 16, 0, stream>>>(
        (const __bf16*)A, (const __bf16*)B, C, M, N, K);
}

}
