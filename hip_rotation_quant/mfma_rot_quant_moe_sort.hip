// MFMA Rotation + MXFP4 Quant + MoE Sorted Scale Scatter
// v_mfma_f32_16x16x32_bf16 + v_cvt_scalef32_pk_fp4_f32
// Target: MI355X (gfx950, CDNA4)
//
// MFMA 16x16x32 BF16 layout:
//   A: lane%16 = M-row, lane/16 = k_group, 8 bf16 per lane
//   B: lane%16 = N-col,  lane/16 = k_group, 8 bf16 per lane
//   C: lane%16 = col,    row = (lane/16)*4 + i (i=0..3)
//
// Wave tiling: 2x2 waves (256 threads)
//   wave_m = wave_id/2 (0,1) -> 16 rows each -> 32 rows total
//   wave_n = wave_id%2 (0,1) -> 64 cols each -> 128 cols total

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

constexpr int RS = 128;
constexpr int QG = 32;
constexpr int NUM_QG = RS / QG;   // 4
constexpr int BLOCK_M = 32;
constexpr int THREADS = 256;
constexpr int TILE_W = 16;

using float4_t = __attribute__((ext_vector_type(4))) float;
using bf16x4_t = __attribute__((ext_vector_type(4))) __bf16;
using bf16x8_t = __attribute__((ext_vector_type(8))) __bf16;

template <int N_I, int TILE_N>
__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void mfma_rot_quant_moe_sort_kernel(
    const __bf16* __restrict__ x,
    const __bf16* __restrict__ rotation,
    uint8_t* __restrict__ fp4_out,
    uint8_t* __restrict__ sorted_scale_out,
    const int* __restrict__ sorted_ids,
    const int* __restrict__ num_valid_ids,
    int M, int K, int token_num, int m_o,
    int stride_x, int stride_fp4,
    int stride_sc_m, int stride_sc_n)
{
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * BLOCK_M;
    const int col_start = blockIdx.y * RS;
    const int lane = tid % 64;
    const int wave_m = (tid / 64) / 2;
    const int wave_n = (tid / 64) % 2;
    const int k_group = lane / 16;
    const int lane16 = lane % 16;

    // ---- Load rotation matrix to LDS ----
    // Simple row-major layout: R_lds[k][n] = rotation[k * RS + n]
    __shared__ __bf16 R_lds[RS][RS]; // 128 x 128 x 2B = 32KB

    for (int i = tid; i < RS * RS; i += THREADS) {
        int k = i / RS;
        int n = i % RS;
        R_lds[k][n] = rotation[k * RS + n];
    }

    __shared__ uint8_t scale_lds[BLOCK_M][NUM_QG];
    __syncthreads();

    // ---- MFMA rotation matmul with scalar B loading ----
    // MFMA B operand: lane16 = N-column, k_group = which 8 K elements
    // b_data[j] = R[k_start + k_group*8 + j][n_col_base + lane16] for j = 0..7
    const int a_row = row_start + wave_m * 16 + lane16;
    const int n_base = wave_n * 64;
    float4_t acc[4] = {};

    #pragma unroll
    for (int kt = 0; kt < 4; kt++) {
        const int k_start = kt * 32;
        const int k_off = k_start + k_group * 8;

        bf16x8_t a_data = {};
        if (a_row < M) {
            auto* a_ptr = x + a_row * stride_x + col_start + k_off;
            a_data = *reinterpret_cast<const bf16x8_t*>(a_ptr);
        }

        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            const int n_col = n_base + nt * TILE_W + lane16;

            // Scalar B loading: read 8 consecutive K values from column n_col
            bf16x8_t b_data;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                b_data[j] = R_lds[k_off + j][n_col];
            }

            acc[nt] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a_data, b_data, acc[nt], 0, 0, 0);
        }
    }

    // ---- MXFP4 quantization ----
    // Output layout: acc[nt][i] -> result[wave_m*16 + (lane/16)*4 + i, n_base + nt*16 + lane16]
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int row_local = wave_m * 16 + (lane / 16) * 4 + i;
        int global_row = row_start + row_local;
        if (global_row >= M) continue;

        #pragma unroll
        for (int qg = 0; qg < 2; qg++) {
            float v0 = acc[qg * 2][i];
            float v1 = acc[qg * 2 + 1][i];

            float amax = fmaxf(fabsf(v0), fabsf(v1));
            amax = fmaxf(amax, __shfl_xor(amax, 1));
            amax = fmaxf(amax, __shfl_xor(amax, 2));
            amax = fmaxf(amax, __shfl_xor(amax, 4));
            amax = fmaxf(amax, __shfl_xor(amax, 8));

            uint32_t u32 = __float_as_uint(amax);
            u32 = (u32 + 0x200000) & 0xFF800000;
            uint32_t raw_exp = (u32 >> 23) & 0xFF;
            uint32_t e8m0 = max(raw_exp, 2u) - 2u;
            float hw_scale = __uint_as_float(e8m0 << 23);

            const int out_base = global_row * stride_fp4 + blockIdx.y * (RS / 2);

            float odd0 = __shfl(v0, lane | 1);
            if ((lane16 & 1) == 0) {
                uint32_t pk;
                asm volatile("v_cvt_scalef32_pk_fp4_f32 %0, %1, %2, %3"
                    : "=v"(pk) : "v"(v0), "v"(odd0), "v"(hw_scale));
                fp4_out[out_base + (n_base + qg * 32 + lane16) / 2] = (uint8_t)pk;
            }

            float odd1 = __shfl(v1, lane | 1);
            if ((lane16 & 1) == 0) {
                uint32_t pk;
                asm volatile("v_cvt_scalef32_pk_fp4_f32 %0, %1, %2, %3"
                    : "=v"(pk) : "v"(v1), "v"(odd1), "v"(hw_scale));
                fp4_out[out_base + (n_base + qg * 32 + 16 + lane16) / 2] = (uint8_t)pk;
            }

            if (lane16 == 0) {
                int qg_idx = (n_base + qg * 32) / QG;
                if (qg_idx < NUM_QG && row_local < BLOCK_M)
                    scale_lds[row_local][qg_idx] = (uint8_t)e8m0;
            }
        }
    }
    __syncthreads();

    // ---- MoE sorted scale scatter ----
    const int num_valid = *num_valid_ids;
    const int sc_col_base = blockIdx.y * NUM_QG;

    for (int q = tid; q < num_valid; q += THREADS) {
        int tok = sorted_ids[q] & 0xFFFFFF;
        if (tok < row_start || tok >= row_start + BLOCK_M || tok >= token_num)
            continue;

        #pragma unroll
        for (int ci = 0; ci < NUM_QG; ci++) {
            int col = sc_col_base + ci;
            if (col >= N_I) continue;

            uint8_t val = scale_lds[tok - row_start][ci];

            int q_in = q & 31;
            int flat = (((((q >> 5) * TILE_N + (col >> 3)) * 4
                        + (col & 3)) * 16 + (q_in & 15)) * 4
                        + (q_in >> 4) + (((col & 7) >> 2) << 1));

            sorted_scale_out[(flat / N_I) * stride_sc_m
                           + (flat % N_I) * stride_sc_n] = val;
        }
    }
}

extern "C" void launch_mfma_rot_quant_moe_sort(
    uint8_t* fp4_out, uint8_t* sorted_scale_out,
    const void* x, const void* rot,
    const int* sorted_ids, const int* num_valid_ids,
    int M, int K, int token_num, int m_o, int m_pad,
    int stride_x, int stride_fp4,
    int stride_sc_m, int stride_sc_n,
    int n_i, hipStream_t stream)
{
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M, K / RS);
    dim3 block(THREADS);

    if (n_i == 64) {
        mfma_rot_quant_moe_sort_kernel<64, 8><<<grid, block, 0, stream>>>(
            (const __bf16*)x, (const __bf16*)rot, fp4_out, sorted_scale_out,
            sorted_ids, num_valid_ids,
            M, K, token_num, m_o, stride_x, stride_fp4, stride_sc_m, stride_sc_n);
    } else if (n_i == 224) {
        mfma_rot_quant_moe_sort_kernel<224, 28><<<grid, block, 0, stream>>>(
            (const __bf16*)x, (const __bf16*)rot, fp4_out, sorted_scale_out,
            sorted_ids, num_valid_ids,
            M, K, token_num, m_o, stride_x, stride_fp4, stride_sc_m, stride_sc_n);
    } else if (n_i == 256) {
        mfma_rot_quant_moe_sort_kernel<256, 32><<<grid, block, 0, stream>>>(
            (const __bf16*)x, (const __bf16*)rot, fp4_out, sorted_scale_out,
            sorted_ids, num_valid_ids,
            M, K, token_num, m_o, stride_x, stride_fp4, stride_sc_m, stride_sc_n);
    }
}
