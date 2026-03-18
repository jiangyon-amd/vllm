// Unified Fused Rotation + MXFP4 Quantization — Dense + MoE in one kernel
// Target: MI355X (gfx950, CDNA4)
//
// Dense mode: rotation → FP4 + raw/shuffled scales
// MoE mode:   rotation → FP4 + sorted scatter scales (via sorted_ids)
//
// Build: hipcc -shared -fPIC -O2 --offload-arch=gfx950 -o unified_fused_rot_quant.so unified_fused_rot_quant.hip

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

constexpr int RS = 128;
constexpr int QG = 32;
constexpr int NUM_QG = RS / QG;
constexpr int BLOCK_M = 32;
constexpr int THREADS = 256;

using float4_t = __attribute__((ext_vector_type(4))) float;
using bf16x8_t = __attribute__((ext_vector_type(8))) __bf16;

// Scale output mode
constexpr int SCALE_RAW = 0;
constexpr int SCALE_SHUFFLE = 1;
constexpr int SCALE_MOE_SORT = 2;

__device__ __forceinline__ float fp4_scale_func(float absMax) {
    uint32_t u32 = __float_as_uint(absMax);
    uint32_t exponent = (u32 >> 23) & 0xFF;
    if (exponent == 0xFF) return __uint_as_float(exponent << 23);
    if ((u32 & 0x400000) && ((u32 & 0x200000) || (u32 & 0x1FFFFF) || exponent))
        exponent += 1;
    return __uint_as_float(exponent << 23);
}

__device__ __forceinline__ int shuffle_scale_idx(int sn_pad, int m_local, int col) {
    int i1 = m_local / 16, i2 = m_local % 16;
    int i3 = col / 8, i4 = (col % 8) / 4, i5 = col % 4;
    int flat = i3 * 256 + i5 * 64 + i2 * 4 + i4 * 2 + i1;
    return (flat / sn_pad) * sn_pad + (flat % sn_pad);
}

template <int SCALE_MODE, int N_I = 0, int TILE_N = 0>
__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void unified_fused_rot_quant_kernel(
    const __bf16* __restrict__ x,
    const __bf16* __restrict__ rotation,
    uint8_t* __restrict__ fp4_out,
    uint8_t* __restrict__ scale_out,
    // MoE-only params (ignored when SCALE_MODE != SCALE_MOE_SORT)
    const int* __restrict__ sorted_ids,
    const int* __restrict__ num_valid_ids,
    int M, int K, int token_num, int m_o, int m_pad,
    int stride_x, int stride_fp4,
    int stride_sc_m, int stride_sc_n,
    int sn_pad)
{
    int tid = threadIdx.x;
    int bm = blockIdx.x;
    int block_rot = blockIdx.y;
    int row_start = bm * BLOCK_M;
    int col_start = block_rot * RS;

    int wave_id = tid / 64;
    int lane_id = tid % 64;
    int wave_m = wave_id / 2;
    int wave_n = wave_id % 2;

    // ======== Phase 1: Load rotation (vectorized) ========
    __shared__ __bf16 R_lds[RS][RS];

    #pragma unroll
    for (int iter = 0; iter < 8; iter++) {
        int vec_idx = iter * 256 + tid;
        int row = vec_idx / 16;
        int col = (vec_idx % 16) * 8;
        bf16x8_t val = *reinterpret_cast<const bf16x8_t*>(&rotation[row * RS + col]);
        *reinterpret_cast<bf16x8_t*>(&R_lds[row][col]) = val;
    }

    __shared__ uint8_t scale_lds[BLOCK_M][NUM_QG];

    __syncthreads();

    // ======== Phase 2: MFMA rotation matmul ========
    int k_group = lane_id / 16;
    int lane_16 = lane_id % 16;
    int a_global_row = row_start + wave_m * 16 + lane_16;
    int n_base = wave_n * 64;

    float4_t acc[4] = {};

    #pragma unroll
    for (int kt = 0; kt < 4; kt++) {
        int k_start = kt * 32;
        int a_k_base = k_start + k_group * 8;

        bf16x8_t a_data;
        if (a_global_row < M) {
            a_data = *reinterpret_cast<const bf16x8_t*>(
                x + a_global_row * stride_x + col_start + a_k_base);
        } else {
            a_data = {};
        }

        #pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            int n_col = n_base + nt * 16 + lane_16;
            bf16x8_t b_data;
            #pragma unroll
            for (int j = 0; j < 8; j++)
                b_data[j] = R_lds[k_start + k_group * 8 + j][n_col];

            acc[nt] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a_data, b_data, acc[nt], 0, 0, 0);
        }
    }

    // ======== Phase 3: FP4 quantization + direct FP4 store ========
    int c_m_group = lane_id / 16;
    int c_col = lane_id % 16;
    int fp4_base = block_rot * (RS / 2);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int row_local = wave_m * 16 + c_m_group * 4 + i;
        int global_row = row_start + row_local;
        if (global_row >= M) continue;

        #pragma unroll
        for (int qg = 0; qg < 2; qg++) {
            float v0 = __bfloat162float(__float2bfloat16(acc[qg * 2][i]));
            float v1 = __bfloat162float(__float2bfloat16(acc[qg * 2 + 1][i]));

            float local_amax = fmaxf(fabsf(v0), fabsf(v1));
            local_amax = fmaxf(local_amax, __shfl_xor(local_amax, 1));
            local_amax = fmaxf(local_amax, __shfl_xor(local_amax, 2));
            local_amax = fmaxf(local_amax, __shfl_xor(local_amax, 4));
            local_amax = fmaxf(local_amax, __shfl_xor(local_amax, 8));

            float scale = fp4_scale_func(local_amax) * 0.25f;
            uint8_t e8m0 = (__float_as_uint(scale) >> 23) & 0xFF;

            float odd = __shfl(v0, lane_id | 1);
            if ((c_col & 1) == 0) {
                uint32_t pk;
                asm volatile("v_cvt_scalef32_pk_fp4_f32 %0, %1, %2, %3"
                    : "=v"(pk) : "v"(v0), "v"(odd), "v"(scale));
                fp4_out[global_row * stride_fp4 + fp4_base + (n_base + qg * 32) / 2 + c_col / 2] = (uint8_t)pk;
            }

            float odd1 = __shfl(v1, lane_id | 1);
            if ((c_col & 1) == 0) {
                uint32_t pk;
                asm volatile("v_cvt_scalef32_pk_fp4_f32 %0, %1, %2, %3"
                    : "=v"(pk) : "v"(v1), "v"(odd1), "v"(scale));
                fp4_out[global_row * stride_fp4 + fp4_base + (n_base + qg * 32 + 16) / 2 + c_col / 2] = (uint8_t)pk;
            }

            if (c_col == 0)
                scale_lds[row_local][(n_base + qg * 32) / QG] = e8m0;
        }
    }
    __syncthreads();

    // ======== Phase 4: Scale output (mode-dependent) ========

    if constexpr (SCALE_MODE == SCALE_RAW) {
        // Dense raw: scale_out[global_m, orig_col] = e8m0
        if (tid < BLOCK_M * NUM_QG) {
            int m_local = tid / NUM_QG;
            int qg = tid % NUM_QG;
            int global_m = row_start + m_local;
            if (global_m < M)
                scale_out[global_m * stride_sc_m + block_rot * NUM_QG + qg] = scale_lds[m_local][qg];
        }
    }
    else if constexpr (SCALE_MODE == SCALE_SHUFFLE) {
        // Dense shuffle: scale_out[shuffled_row, shuffled_col] = e8m0
        if (tid < BLOCK_M * NUM_QG) {
            int m_local = tid / NUM_QG;
            int qg = tid % NUM_QG;
            int global_m = row_start + m_local;
            uint8_t e8m0_val = (global_m < M) ? scale_lds[m_local][qg] : 0;
            int orig_col = block_rot * NUM_QG + qg;
            int flat = shuffle_scale_idx(sn_pad, m_local, orig_col);
            int sh_row = flat / sn_pad + row_start;
            int sh_col = flat % sn_pad;
            if (sh_row < row_start + BLOCK_M)
                scale_out[sh_row * stride_sc_m + sh_col] = e8m0_val;
        }
    }
    else if constexpr (SCALE_MODE == SCALE_MOE_SORT) {
        // MoE sorted scatter: scale_out[sorted_5d_offset] = e8m0
        int num_valid = *num_valid_ids;
        int scale_col_base = block_rot * NUM_QG;

        for (int q = tid; q < m_o && q < num_valid; q += THREADS) {
            int sid = sorted_ids[q];
            int tok = sid & 0xFFFFFF;
            if (tok >= row_start && tok < (row_start + BLOCK_M) && tok < token_num) {
                int tok_local = tok - row_start;
                #pragma unroll
                for (int ci = 0; ci < NUM_QG; ci++) {
                    int col = scale_col_base + ci;
                    if (col >= N_I) continue;
                    uint8_t e8m0_val = scale_lds[tok_local][ci];
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
                    scale_out[(flat / N_I) * stride_sc_m + (flat % N_I) * stride_sc_n] = e8m0_val;
                }
            }
        }
    }
}

// ======== Launchers ========

extern "C" void launch_unified_dense_raw(
    uint8_t* fp4_out, uint8_t* scale_out,
    const void* x, const void* rot,
    int M, int K,
    int stride_x, int stride_fp4,
    int stride_sc, int sn_pad,
    hipStream_t stream)
{
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M, K / RS);
    unified_fused_rot_quant_kernel<SCALE_RAW><<<grid, 256, 0, stream>>>(
        (const __bf16*)x, (const __bf16*)rot, fp4_out, scale_out,
        nullptr, nullptr,
        M, K, 0, 0, 0,
        stride_x, stride_fp4, stride_sc, 1, sn_pad);
}

extern "C" void launch_unified_dense_shuffle(
    uint8_t* fp4_out, uint8_t* scale_out,
    const void* x, const void* rot,
    int M, int K,
    int stride_x, int stride_fp4,
    int stride_sc, int sn_pad,
    hipStream_t stream)
{
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M, K / RS);
    unified_fused_rot_quant_kernel<SCALE_SHUFFLE><<<grid, 256, 0, stream>>>(
        (const __bf16*)x, (const __bf16*)rot, fp4_out, scale_out,
        nullptr, nullptr,
        M, K, 0, 0, 0,
        stride_x, stride_fp4, stride_sc, 1, sn_pad);
}

extern "C" void launch_unified_moe_sort(
    uint8_t* fp4_out, uint8_t* scale_out,
    const void* x, const void* rot,
    const int* sorted_ids, const int* num_valid_ids,
    int M, int K, int token_num, int m_o, int m_pad,
    int stride_x, int stride_fp4,
    int stride_sc_m, int stride_sc_n,
    int n_i, hipStream_t stream)
{
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M, K / RS);
    if (n_i == 64)
        unified_fused_rot_quant_kernel<SCALE_MOE_SORT, 64, 8><<<grid, 256, 0, stream>>>(
            (const __bf16*)x, (const __bf16*)rot, fp4_out, scale_out,
            sorted_ids, num_valid_ids,
            M, K, token_num, m_o, m_pad,
            stride_x, stride_fp4, stride_sc_m, stride_sc_n, 0);
    else if (n_i == 224)
        unified_fused_rot_quant_kernel<SCALE_MOE_SORT, 224, 28><<<grid, 256, 0, stream>>>(
            (const __bf16*)x, (const __bf16*)rot, fp4_out, scale_out,
            sorted_ids, num_valid_ids,
            M, K, token_num, m_o, m_pad,
            stride_x, stride_fp4, stride_sc_m, stride_sc_n, 0);
    else if (n_i == 256)
        unified_fused_rot_quant_kernel<SCALE_MOE_SORT, 256, 32><<<grid, 256, 0, stream>>>(
            (const __bf16*)x, (const __bf16*)rot, fp4_out, scale_out,
            sorted_ids, num_valid_ids,
            M, K, token_num, m_o, m_pad,
            stride_x, stride_fp4, stride_sc_m, stride_sc_n, 0);
}
