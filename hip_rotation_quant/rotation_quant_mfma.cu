// Fused Rotation + MXFP4 Quantization HIP Kernel with MFMA
//
// Uses v_mfma_f32_16x16x32_f16 for rotation matmul (LDS-based)
// Uses v_cvt_scalef32_pk_fp4_f32 for fp4 quantization
// Scale: 0x400000 rounding — aligned with Dense Gluon v2 path
//
// Target: MI355X (gfx950, CDNA4)
// Supports: rotation_size = 32, 64, 128 (template parameter)

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

using fp16x8 = __attribute__((ext_vector_type(8))) _Float16;
using fp16x4 = __attribute__((ext_vector_type(4))) _Float16;
using fp32x4 = __attribute__((ext_vector_type(4))) float;

template<int RS, int BLOCK_M>
__global__ __launch_bounds__(256)
void fused_rotation_quant_mfma_kernel(
    uint8_t* __restrict__ out_fp4,       // [M, K/2]
    uint8_t* __restrict__ out_scale,     // [sm_pad, sn_pad] or [M, K/32]
    const __bf16* __restrict__ x,        // [M, K] bf16
    const __bf16* __restrict__ rot,      // [RS, RS] bf16
    int M, int K,
    int stride_fp4_m,                    // K/2
    int stride_sc_m,                     // sn_padded
    int sn_padded,
    int shuffle_scales                   // 0 or 1
) {
    constexpr int THREADS = 256;
    constexpr int WAVES = 4;
    constexpr int N_TILES = RS / 16;
    constexpr int K_ITERS = RS / 16;  // MFMA 16x16x16: each iter covers K=16
    constexpr int M_TILES = BLOCK_M / 16;
    constexpr int QG = 32;
    constexpr int NUM_QG = RS / QG;

    const int bid_m = blockIdx.x, bid_k = blockIdx.y, tid = threadIdx.x;
    const int wave_id = tid / 64, lane_id = tid % 64;
    const int m_base = bid_m * BLOCK_M, k_base = bid_k * RS;

    // === LDS ===
    __shared__ __bf16 rot_lds[RS * RS];
    __shared__ __bf16 a_lds[BLOCK_M * RS];
    __shared__ float rotated[BLOCK_M * RS];

    // === Step 1: Load rotation & A to LDS (64-bit vector loads) ===
    {
        uint64_t* dst = reinterpret_cast<uint64_t*>(rot_lds);
        const uint64_t* src = reinterpret_cast<const uint64_t*>(rot);
        #pragma unroll 4
        for (int i = tid; i < RS * RS / 4; i += THREADS) dst[i] = src[i];
    }
    {
        #pragma unroll 4
        for (int i = tid; i < BLOCK_M * RS / 4; i += THREADS) {
            int elem = i * 4;
            int ml = elem / RS, kl = elem % RS;
            int gm = m_base + ml;
            if (gm < M)
                reinterpret_cast<uint64_t*>(a_lds)[i] =
                    *reinterpret_cast<const uint64_t*>(&x[gm * K + k_base + kl]);
            else
                reinterpret_cast<uint64_t*>(a_lds)[i] = 0ULL;
        }
    }
    __syncthreads();

    // === Step 2: MFMA Rotation (preload B, unrolled) ===
    // MFMA 16x16x16_f16: matches Gluon v2's k_width=4 accumulation pattern
    // Each lane loads 4 bf16→fp16 values (not 8), K step = 16 (not 32)
    // Lane mapping: A row = lane%16, A k_group = (lane/16)*4, 4 groups of 4
    for (int n_tile = wave_id; n_tile < N_TILES; n_tile += WAVES) {
        fp16x4 b_vecs[K_ITERS];
        const int b_col = n_tile * 16 + (lane_id % 16);
        const int b_k_group = (lane_id / 16) * 4;

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            const int b_k_base = ki * 16 + b_k_group;
            #pragma unroll
            for (int j = 0; j < 4; j++)
                b_vecs[ki][j] = (_Float16)__bfloat162float(rot_lds[(b_k_base + j) * RS + b_col]);
        }

        #pragma unroll
        for (int m_tile = 0; m_tile < M_TILES; m_tile++) {
            fp32x4 acc = {0, 0, 0, 0};
            const int a_row = m_tile * 16 + (lane_id % 16);
            const int a_k_group = (lane_id / 16) * 4;

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++) {
                fp16x4 a_vec;
                const __bf16* a_ptr = &a_lds[a_row * RS + ki * 16 + a_k_group];
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    a_vec[j] = (_Float16)__bfloat162float(a_ptr[j]);

                acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_vec, b_vecs[ki], acc, 0, 0, 0);
            }

            const int c_col = n_tile * 16 + (lane_id % 16);
            const int c_row = m_tile * 16 + (lane_id / 16) * 4;
            #pragma unroll
            for (int i = 0; i < 4; i++)
                // bf16 truncation to match Gluon v2: acc.to(bf16).to(fp32)
                rotated[(c_row + i) * RS + c_col] = __bfloat162float(__float2bfloat16(acc[i]));
        }
    }
    __syncthreads();

    // === Step 3: Quantization (0x400000 rounding — Gluon v2 aligned) ===
    __shared__ float grp_amax[BLOCK_M][NUM_QG];
    __shared__ uint8_t grp_e8m0[BLOCK_M][NUM_QG];
    __shared__ float grp_scale[BLOCK_M][NUM_QG];

    #pragma unroll
    for (int i = tid; i < BLOCK_M * NUM_QG; i += THREADS)
        grp_amax[i / NUM_QG][i % NUM_QG] = 0.0f;
    __syncthreads();

    #pragma unroll 4
    for (int i = tid; i < BLOCK_M * RS; i += THREADS) {
        int ml = i / RS, kl = i % RS, gm = m_base + ml;
        if (gm < M)
            atomicMax(reinterpret_cast<int*>(&grp_amax[ml][kl / QG]),
                      __float_as_int(fabsf(rotated[i])));
    }
    __syncthreads();

    // Scale: 0x400000 rounding + e8m0 = max(raw_exp, 2) - 2 (Gluon v2 style)
    for (int i = tid; i < BLOCK_M * NUM_QG; i += THREADS) {
        int ml = i / NUM_QG, qg = i % NUM_QG;
        float amax = __int_as_float(*reinterpret_cast<int*>(&grp_amax[ml][qg]));

        uint32_t au = __float_as_uint(amax);
        uint32_t rounded = (au + 0x400000u) & 0xFF800000u;
        uint32_t raw_exp = (rounded >> 23) & 0xFF;
        uint32_t e8m0 = (raw_exp > 2 ? raw_exp : 2) - 2;

        grp_e8m0[ml][qg] = (uint8_t)e8m0;
        // hw_scale for v_cvt: 2^e8m0 (instruction does src / scale)
        uint32_t hw_scale_u32 = e8m0 << 23;
        grp_scale[ml][qg] = __uint_as_float(hw_scale_u32);
    }
    __syncthreads();

    // FP4 conversion via v_cvt_scalef32_pk_fp4_f32
    #pragma unroll 4
    for (int i = tid; i < BLOCK_M * (RS / 2); i += THREADS) {
        int ml = i / (RS / 2), pi = i % (RS / 2), kl = pi * 2;
        int gm = m_base + ml;
        if (gm >= M) continue;

        float v0 = rotated[ml * RS + kl];
        float v1 = rotated[ml * RS + kl + 1];
        float sc = grp_scale[ml][kl / QG];

        uint32_t packed;
        asm volatile("v_cvt_scalef32_pk_fp4_f32 %0, %1, %2, %3"
                     : "=v"(packed) : "v"(v0), "v"(v1), "v"(sc));

        out_fp4[gm * stride_fp4_m + k_base / 2 + pi] = (uint8_t)packed;
    }

    // Store scales
    for (int i = tid; i < BLOCK_M * NUM_QG; i += THREADS) {
        int ml = i / NUM_QG, qg = i % NUM_QG;
        int gm = m_base + ml;
        if (gm >= M) continue;

        uint8_t e8m0_val = grp_e8m0[ml][qg];
        int orig_col = (k_base / QG) + qg;

        if (shuffle_scales) {
            // Shuffle for Dense GEMM (e8m0_shuffle format)
            int i1 = ml / 16, i2 = ml % 16;
            int i3 = orig_col / 8, i4 = (orig_col % 8) / 4, i5 = orig_col % 4;
            int flat = i3 * 256 + i5 * 64 + i2 * 4 + i4 * 2 + i1;
            int sh_row = flat / sn_padded + m_base;
            int sh_col = flat % sn_padded;
            if (sh_row < m_base + BLOCK_M)
                out_scale[sh_row * stride_sc_m + sh_col] = e8m0_val;
        } else {
            out_scale[gm * stride_sc_m + orig_col] = e8m0_val;
        }
    }
}

#ifdef BUILD_TORCH_EXT
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>

torch::Tensor fused_rotation_quant_mfma(
    torch::Tensor x, torch::Tensor rotation,
    torch::Tensor fp4_out, torch::Tensor scale_out,
    int rotation_size, int shuffle_scales
) {
    int M = x.size(0), K = x.size(1);
    int sn_padded = scale_out.size(1);
    constexpr int BM = 32;

    auto stream = at::hip::getCurrentHIPStream().stream();
    dim3 grid((M + BM - 1) / BM, K / rotation_size);

    if (rotation_size == 128)
        fused_rotation_quant_mfma_kernel<128, BM><<<grid, 256, 0, stream>>>(
            fp4_out.data_ptr<uint8_t>(), scale_out.data_ptr<uint8_t>(),
            (const __bf16*)x.data_ptr(), (const __bf16*)rotation.data_ptr(),
            M, K, K/2, sn_padded, sn_padded, shuffle_scales);
    else if (rotation_size == 64)
        fused_rotation_quant_mfma_kernel<64, BM><<<grid, 256, 0, stream>>>(
            fp4_out.data_ptr<uint8_t>(), scale_out.data_ptr<uint8_t>(),
            (const __bf16*)x.data_ptr(), (const __bf16*)rotation.data_ptr(),
            M, K, K/2, sn_padded, sn_padded, shuffle_scales);
    else
        fused_rotation_quant_mfma_kernel<32, BM><<<grid, 256, 0, stream>>>(
            fp4_out.data_ptr<uint8_t>(), scale_out.data_ptr<uint8_t>(),
            (const __bf16*)x.data_ptr(), (const __bf16*)rotation.data_ptr(),
            M, K, K/2, sn_padded, sn_padded, shuffle_scales);

    return fp4_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rotation_quant_mfma", &fused_rotation_quant_mfma,
          "Fused rotation + MXFP4 quant with MFMA (HIP, Gluon v2 aligned)");
}
#endif
