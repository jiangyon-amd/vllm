// Fused Rotation + MXFP4 Quant + Sorted Scale Scatter — M>1 HIP kernel
// Eliminates Triton overhead + temp row by using LDS for scale relay
// Target: MI355X (gfx950, CDNA4)

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

constexpr int RS = 128;
constexpr int QG = 32;
constexpr int NUM_QG = RS / QG;  // 4
constexpr int HALF_QG = QG / 2;  // 16
constexpr int WAVEFRONT = 64;
constexpr int BLOCK_M = 32;

__device__ __forceinline__ float warp_max(float val) {
    #pragma unroll
    for (int mask = WAVEFRONT / 2; mask >= 1; mask /= 2) {
        val = fmaxf(val, __shfl_xor(val, mask));
    }
    return val;
}

template <int N_I, int TILE_N>
__global__ __launch_bounds__(256)
void fused_rot_quant_sort_m_gt1_kernel(
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_sorted_scale,
    const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ rot,
    const int* __restrict__ sorted_ids,
    const int* __restrict__ num_valid_ids,
    int M, int K, int token_num, int m_o, int m_pad,
    int stride_x_m, int stride_fp4_m,
    int stride_sc_m, int stride_sc_n)
{
    int pid_m = blockIdx.x;
    int pid_k = blockIdx.y;
    int tid = threadIdx.x;
    int m_base = pid_m * BLOCK_M;
    int col_base = pid_k * RS;

    // Load rotation matrix to LDS (32KB, shared across all rows in this block)
    __shared__ float R_lds[RS][RS + 1]; // +1 padding for 64-bank conflict avoidance
    for (int i = tid; i < RS * RS; i += 256) {
        int r = i / RS, c = i % RS;
        R_lds[r][c] = __bfloat162float(rot[r * RS + c]);
    }

    // LDS for e8m0 scales: BLOCK_M × NUM_QG = 32 × 4 = 128 bytes
    __shared__ uint8_t scale_lds[BLOCK_M][NUM_QG];

    __syncthreads();

    // Each thread handles multiple (m, k) output positions
    // 256 threads, BLOCK_M * RS = 32 * 128 = 4096 elements → 16 per thread
    float acc[16];
    
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int flat_idx = tid + i * 256;
        int m_local = flat_idx / RS;
        int k_out = flat_idx % RS;
        int global_m = m_base + m_local;
        
        if (m_local >= BLOCK_M || global_m >= M) {
            acc[i] = 0.0f;
            continue;
        }
        
        // Dot product: x[global_m, col_base:col_base+RS] · R[:, k_out]
        const __hip_bfloat16* x_row = x + global_m * stride_x_m + col_base;
        float sum = 0.0f;
        #pragma unroll 16
        for (int j = 0; j < RS; j += 8) {
            sum += __bfloat162float(x_row[j])   * R_lds[j][k_out]
                 + __bfloat162float(x_row[j+1]) * R_lds[j+1][k_out]
                 + __bfloat162float(x_row[j+2]) * R_lds[j+2][k_out]
                 + __bfloat162float(x_row[j+3]) * R_lds[j+3][k_out]
                 + __bfloat162float(x_row[j+4]) * R_lds[j+4][k_out]
                 + __bfloat162float(x_row[j+5]) * R_lds[j+5][k_out]
                 + __bfloat162float(x_row[j+6]) * R_lds[j+6][k_out]
                 + __bfloat162float(x_row[j+7]) * R_lds[j+7][k_out];
        }
        // BF16 truncation to match MFMA precision
        sum = __bfloat162float(__float2bfloat16(sum));
        acc[i] = sum;
    }

    // Store all acc to LDS for quantization (need sequential access by quant group)
    __shared__ float acc_lds[BLOCK_M][RS];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int flat_idx = tid + i * 256;
        int m_local = flat_idx / RS;
        int k_out = flat_idx % RS;
        if (m_local < BLOCK_M) {
            acc_lds[m_local][k_out] = acc[i];
        }
    }
    __syncthreads();

    // Quantize: each thread handles one (m_local, group) pair
    // 32 rows × 4 groups = 128 pairs, 256 threads → ~2 pairs per thread
    if (tid < BLOCK_M * NUM_QG) {
        int m_local = tid / NUM_QG;
        int g = tid % NUM_QG;
        int global_m = m_base + m_local;
        
        if (global_m < M) {
            // Compute amax for this quant group
            float amax = 0.0f;
            #pragma unroll
            for (int j = 0; j < QG; j++) {
                amax = fmaxf(amax, fabsf(acc_lds[m_local][g * QG + j]));
            }
            
            // E8M0 scale (0x400000 rounding — matches gluon_v2_kw8 reference)
            uint32_t u32 = __float_as_uint(amax);
            u32 = (u32 + 0x400000) & 0xFF800000;
            uint32_t raw_exp = (u32 >> 23) & 0xFF;
            uint32_t e8m0 = max(raw_exp, 2u) - 2u;
            float hw_scale = __uint_as_float(e8m0 << 23);
            
            // Store e8m0 to LDS (for scatter later — NO global temp row!)
            scale_lds[m_local][g] = (uint8_t)e8m0;
            
            // v_cvt_scalef32_pk_fp4_f32: quantize pairs
            int fp4_base = pid_k * (RS / 2) + g * HALF_QG;
            #pragma unroll
            for (int p = 0; p < HALF_QG; p++) {
                float even_val = acc_lds[m_local][g * QG + p * 2];
                float odd_val  = acc_lds[m_local][g * QG + p * 2 + 1];
                uint32_t packed;
                asm volatile("v_cvt_scalef32_pk_fp4_f32 %0, %1, %2, %3"
                             : "=v"(packed)
                             : "v"(even_val), "v"(odd_val), "v"(hw_scale));
                out_fp4[global_m * stride_fp4_m + fp4_base + p] = (uint8_t)(packed & 0xFF);
            }
        }
    }
    __syncthreads();

    // Scatter sorted scales from LDS (no global temp row!)
    int num_valid = *num_valid_ids;
    int scale_col_base = pid_k * NUM_QG;
    
    // Each thread processes multiple sorted_ids entries
    for (int q = tid; q < m_o && q < num_valid; q += 256) {
        int sid = sorted_ids[q];
        int tok = sid & 0xFFFFFF;
        
        // Only process if token is in this block's row range
        if (tok >= m_base && tok < (m_base + BLOCK_M) && tok < token_num) {
            int tok_local = tok - m_base;
            
            #pragma unroll
            for (int ci = 0; ci < NUM_QG; ci++) {
                int col = scale_col_base + ci;
                if (col >= N_I) continue;
                
                uint8_t e8m0_val = scale_lds[tok_local][ci];
                
                // Compute shuffled scale destination
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

// Template instantiations for common N_I values
extern "C" void launch_fused_rot_quant_sort_m_gt1(
    uint8_t* out_fp4,
    uint8_t* out_sorted_scale,
    const void* x,
    const void* rot,
    const int* sorted_ids,
    const int* num_valid_ids,
    int M, int K, int token_num, int m_o, int m_pad,
    int stride_x_m, int stride_fp4_m,
    int stride_sc_m, int stride_sc_n,
    int n_i,
    hipStream_t stream)
{
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M, K / RS);
    dim3 block(256);
    int tile_n = (n_i + 7) / 8;
    
    // Dispatch based on n_i (constexpr template param)
    if (n_i == 64) {
        fused_rot_quant_sort_m_gt1_kernel<64, 8><<<grid, block, 0, stream>>>(
            out_fp4, out_sorted_scale,
            (const __hip_bfloat16*)x, (const __hip_bfloat16*)rot,
            sorted_ids, num_valid_ids,
            M, K, token_num, m_o, m_pad,
            stride_x_m, stride_fp4_m, stride_sc_m, stride_sc_n);
    } else if (n_i == 160) {
        fused_rot_quant_sort_m_gt1_kernel<160, 20><<<grid, block, 0, stream>>>(
            out_fp4, out_sorted_scale,
            (const __hip_bfloat16*)x, (const __hip_bfloat16*)rot,
            sorted_ids, num_valid_ids,
            M, K, token_num, m_o, m_pad,
            stride_x_m, stride_fp4_m, stride_sc_m, stride_sc_n);
    } else if (n_i == 224) {
        fused_rot_quant_sort_m_gt1_kernel<224, 28><<<grid, block, 0, stream>>>(
            out_fp4, out_sorted_scale,
            (const __hip_bfloat16*)x, (const __hip_bfloat16*)rot,
            sorted_ids, num_valid_ids,
            M, K, token_num, m_o, m_pad,
            stride_x_m, stride_fp4_m, stride_sc_m, stride_sc_n);
    } else if (n_i == 256) {
        fused_rot_quant_sort_m_gt1_kernel<256, 32><<<grid, block, 0, stream>>>(
            out_fp4, out_sorted_scale,
            (const __hip_bfloat16*)x, (const __hip_bfloat16*)rot,
            sorted_ids, num_valid_ids,
            M, K, token_num, m_o, m_pad,
            stride_x_m, stride_fp4_m, stride_sc_m, stride_sc_n);
    }
}
