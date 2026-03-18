// Fused Rotation + MXFP4 Quant + MoE Sort — based on rotation_quant_mfma.hip (100% fp4 match)
// Added: LDS-based sorted scale scatter using sorted_ids from moe_sorting
// Target: MI355X (gfx950, CDNA4)

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

constexpr int RS = 128;
constexpr int QG = 32;
constexpr int NUM_QG = RS / QG;   // 4
constexpr int BLOCK_M = 32;
constexpr int WAVEFRONT = 64;
constexpr int NUM_WAVES = 4;
constexpr int THREADS = NUM_WAVES * WAVEFRONT; // 256

__device__ __forceinline__ float fp4_scale_func(float absMax) {
    uint32_t u32 = __float_as_uint(absMax);
    uint32_t exponent = (u32 >> 23) & 0xFF;
    if (exponent == 0xFF) return __uint_as_float(exponent << 23);
    if ((u32 & 0x400000) && ((u32 & 0x200000) || (u32 & 0x1FFFFF) || exponent))
        exponent += 1;
    return __uint_as_float(exponent << 23);
}

template <int N_I, int TILE_N>
__global__ __launch_bounds__(256)
void rotation_quant_moe_sort_kernel(
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_sorted_scale,
    const __bf16* __restrict__ x,
    const __bf16* __restrict__ rot,
    const int* __restrict__ sorted_ids,
    const int* __restrict__ num_valid_ids,
    int M, int K, int token_num, int m_o,
    int stride_x_m, int stride_fp4_m,
    int stride_sc_m, int stride_sc_n)
{
    int block_m = blockIdx.x;
    int block_rot = blockIdx.y;
    int tid = threadIdx.x;
    int m_base = block_m * BLOCK_M;
    int col_base = block_rot * RS;

    // ======== Load rotation to LDS ========
    __shared__ __bf16 R_lds[RS][RS];
    for (int i = tid; i < RS * RS; i += THREADS) {
        R_lds[i / RS][i % RS] = rot[i];
    }
    __syncthreads();

    // ======== Scalar matmul (same as rotation_quant_mfma.hip) ========
    float acc[16];
    for (int i = 0; i < 16; i++) {
        int flat_idx = tid + i * THREADS;
        int m_local = flat_idx / RS;
        int k_out = flat_idx % RS;

        if (m_local >= BLOCK_M) { acc[i] = 0.0f; continue; }
        int global_m = m_base + m_local;
        if (global_m >= M) { acc[i] = 0.0f; continue; }

        float sum = 0.0f;
        const __bf16* x_row = x + global_m * stride_x_m + col_base;
        for (int j = 0; j < RS; j += 4) {
            sum += __bfloat162float(x_row[j])   * __bfloat162float(R_lds[j][k_out])
                 + __bfloat162float(x_row[j+1]) * __bfloat162float(R_lds[j+1][k_out])
                 + __bfloat162float(x_row[j+2]) * __bfloat162float(R_lds[j+2][k_out])
                 + __bfloat162float(x_row[j+3]) * __bfloat162float(R_lds[j+3][k_out]);
        }
        acc[i] = __bfloat162float(__float2bfloat16(sum));
    }

    // ======== Quantization (same as rotation_quant_mfma.hip) ========
    __shared__ float group_amax[BLOCK_M][NUM_QG];
    __shared__ uint8_t group_e8m0[BLOCK_M][NUM_QG];
    __shared__ float group_scale[BLOCK_M][NUM_QG];

    if (tid < BLOCK_M * NUM_QG)
        group_amax[tid / NUM_QG][tid % NUM_QG] = 0.0f;
    __syncthreads();

    for (int i = 0; i < 16; i++) {
        int flat_idx = tid + i * THREADS;
        int m_local = flat_idx / RS;
        int k_out = flat_idx % RS;
        if (m_local >= BLOCK_M) continue;
        atomicMax((int*)&group_amax[m_local][k_out / QG], __float_as_int(fabsf(acc[i])));
    }
    __syncthreads();

    if (tid < BLOCK_M * NUM_QG) {
        int m_local = tid / NUM_QG;
        int qg = tid % NUM_QG;
        float amax = __int_as_float(*(int*)&group_amax[m_local][qg]);
        float scale = fp4_scale_func(amax) * 0.25f;
        group_e8m0[m_local][qg] = (__float_as_uint(scale) >> 23) & 0xFF;
        group_scale[m_local][qg] = scale;
    }
    __syncthreads();

    // Store FP4
    for (int i = 0; i < 16; i += 2) {
        int flat0 = tid + i * THREADS;
        int m0 = flat0 / RS, k0 = flat0 % RS;
        if (m0 >= BLOCK_M || m_base + m0 >= M) continue;
        if (k0 % 2 != 0) continue;

        uint32_t packed;
        asm volatile("v_cvt_scalef32_pk_fp4_f32 %0, %1, %2, %3"
                     : "=v"(packed) : "v"(acc[i]), "v"(acc[i+1]), "v"(group_scale[m0][k0/QG]));
        out_fp4[(m_base + m0) * stride_fp4_m + block_rot * (RS/2) + k0/2] = (uint8_t)packed;
    }

    // ======== MoE Sort Scatter from LDS ========
    int num_valid = *num_valid_ids;
    int scale_col_base = block_rot * NUM_QG;

    for (int q = tid; q < m_o && q < num_valid; q += THREADS) {
        int sid = sorted_ids[q];
        int tok = sid & 0xFFFFFF;

        if (tok >= m_base && tok < (m_base + BLOCK_M) && tok < token_num) {
            int tok_local = tok - m_base;

            #pragma unroll
            for (int ci = 0; ci < NUM_QG; ci++) {
                int col = scale_col_base + ci;
                if (col >= N_I) continue;

                uint8_t e8m0_val = group_e8m0[tok_local][ci];

                int q_in = q & 31;
                int pid_m_out = q >> 5;
                int q_half = q_in >> 4;
                int m_local_sc = q_in & 15;
                int pid_n = col >> 3;
                int n_local = col & 3;
                int half = (col & 7) >> 2;
                int i_val = q_half + (half << 1);

                int flat = ((((pid_m_out * TILE_N + pid_n) * 4 + n_local) * 16
                            + m_local_sc) * 4 + i_val);
                int dst_row = flat / N_I;
                int dst_col = flat % N_I;

                out_sorted_scale[dst_row * stride_sc_m + dst_col * stride_sc_n] = e8m0_val;
            }
        }
    }
}

extern "C" void launch_rotation_quant_moe_sort(
    uint8_t* fp4_out, uint8_t* sorted_scale_out,
    const void* x, const void* rot,
    const int* sorted_ids, const int* num_valid_ids,
    int M, int K, int token_num, int m_o,
    int stride_x_m, int stride_fp4_m,
    int stride_sc_m, int stride_sc_n,
    int n_i, hipStream_t stream)
{
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M, K / RS);
    dim3 block(THREADS);

    if (n_i == 64) {
        rotation_quant_moe_sort_kernel<64, 8><<<grid, block, 0, stream>>>(
            fp4_out, sorted_scale_out, (const __bf16*)x, (const __bf16*)rot,
            sorted_ids, num_valid_ids, M, K, token_num, m_o,
            stride_x_m, stride_fp4_m, stride_sc_m, stride_sc_n);
    } else if (n_i == 128) {
        rotation_quant_moe_sort_kernel<128, 16><<<grid, block, 0, stream>>>(
            fp4_out, sorted_scale_out, (const __bf16*)x, (const __bf16*)rot,
            sorted_ids, num_valid_ids, M, K, token_num, m_o,
            stride_x_m, stride_fp4_m, stride_sc_m, stride_sc_n);
    } else if (n_i == 256) {
        rotation_quant_moe_sort_kernel<256, 32><<<grid, block, 0, stream>>>(
            fp4_out, sorted_scale_out, (const __bf16*)x, (const __bf16*)rot,
            sorted_ids, num_valid_ids, M, K, token_num, m_o,
            stride_x_m, stride_fp4_m, stride_sc_m, stride_sc_n);
    }
}
