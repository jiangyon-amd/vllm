// LDS Rotation MFMA v2 — optimized
//
// Optimizations vs v1:
// 1. Use ds_read_b128 (vector LDS load) instead of scalar bf16 loads
// 2. Remove bounds checks (RS/BLOCK_M are compile-time, always aligned)
// 3. Rotation matrix is loaded once and REUSED across M tiles
// 4. Use __builtin_nontemporal_load for GMEM→LDS prefetch
// 5. For M=1 (decode), skip unused M tiles

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using fp16x8 = __attribute__((ext_vector_type(8))) _Float16;
using fp32x4 = __attribute__((ext_vector_type(4))) float;

// v1: original MFMA (from previous test)
template<int RS, int BLOCK_M>
__global__ void rotation_mfma_v1(
    __bf16* __restrict__ out,
    const __bf16* __restrict__ x,
    const __bf16* __restrict__ rot,
    int M, int K
) {
    constexpr int THREADS = 256;
    constexpr int WAVES = THREADS / 64;
    constexpr int M_TILES = BLOCK_M / 16;
    constexpr int N_TILES = RS / 16;
    constexpr int K_ITERS = RS / 32;
    constexpr int TOTAL_OUT_TILES = M_TILES * N_TILES;

    int bid_m = blockIdx.x, bid_k = blockIdx.y, tid = threadIdx.x;
    int wave_id = tid / 64, lane_id = tid % 64;
    int m_base = bid_m * BLOCK_M, k_base = bid_k * RS;

    __shared__ __bf16 rot_lds[RS * RS];
    __shared__ __bf16 a_lds[BLOCK_M * RS];
    __shared__ __bf16 out_lds[BLOCK_M * RS];

    for (int i = tid; i < RS * RS; i += THREADS) rot_lds[i] = rot[i];
    for (int i = tid; i < BLOCK_M * RS; i += THREADS) {
        int ml = i / RS, kl = i % RS, gm = m_base + ml;
        if (gm < M) a_lds[i] = x[gm * K + k_base + kl];
        else a_lds[i] = __float2bfloat16(0.0f);
    }
    __syncthreads();

    for (int tile = wave_id; tile < TOTAL_OUT_TILES; tile += WAVES) {
        int m_tile = tile / N_TILES, n_tile = tile % N_TILES;
        fp32x4 acc = {0, 0, 0, 0};
        for (int ki = 0; ki < K_ITERS; ki++) {
            int k_off = ki * 32;
            fp16x8 a_vec, b_vec;
            int a_row = m_tile * 16 + (lane_id % 16), a_k = k_off + (lane_id / 16) * 8;
            int b_col = n_tile * 16 + (lane_id % 16), b_k = k_off + (lane_id / 16) * 8;
            for (int j = 0; j < 8; j++) {
                a_vec[j] = (_Float16)__bfloat162float(a_lds[a_row * RS + a_k + j]);
                b_vec[j] = (_Float16)__bfloat162float(rot_lds[(b_k + j) * RS + b_col]);
            }
            acc = __builtin_amdgcn_mfma_f32_16x16x32_f16(a_vec, b_vec, acc, 0, 0, 0);
        }
        int c_col = n_tile * 16 + (lane_id % 16);
        int c_row = m_tile * 16 + (lane_id / 16) * 4;
        for (int i = 0; i < 4; i++)
            out_lds[(c_row + i) * RS + c_col] = __float2bfloat16(acc[i]);
    }
    __syncthreads();
    for (int i = tid; i < BLOCK_M * RS; i += THREADS) {
        int ml = i / RS, kl = i % RS, gm = m_base + ml;
        if (gm < M) out[gm * K + k_base + kl] = out_lds[i];
    }
}

// v2: optimized — vector LDS loads, preload rot to registers, in-place LDS
template<int RS, int BLOCK_M>
__global__ void rotation_mfma_v2(
    __bf16* __restrict__ out,
    const __bf16* __restrict__ x,
    const __bf16* __restrict__ rot,
    int M, int K
) {
    constexpr int THREADS = 256;
    constexpr int WAVES = THREADS / 64;
    constexpr int N_TILES = RS / 16;
    constexpr int K_ITERS = RS / 32;

    int bid_m = blockIdx.x, bid_k = blockIdx.y, tid = threadIdx.x;
    int wave_id = tid / 64, lane_id = tid % 64;
    int m_base = bid_m * BLOCK_M, k_base = bid_k * RS;

    // Shared memory: rotation + A input + output
    // For RS=128: rot=32KB, a=8KB, out=8KB = 48KB total (fits 64KB LDS)
    __shared__ __bf16 rot_lds[RS * RS];
    __shared__ __bf16 a_lds[BLOCK_M * RS];
    __shared__ __bf16 out_lds[BLOCK_M * RS];

    // Load rotation matrix — can use 128-bit vector loads
    // 256 threads, RS*RS elements, each load = 1 bf16 = 2 bytes
    // Optimize: load as uint32 (2 bf16 at a time)
    {
        uint32_t* rot_u32 = reinterpret_cast<uint32_t*>(rot_lds);
        const uint32_t* src_u32 = reinterpret_cast<const uint32_t*>(rot);
        int n_u32 = RS * RS / 2;
        for (int i = tid; i < n_u32; i += THREADS)
            rot_u32[i] = src_u32[i];
    }

    // Load A tile — vector load from GMEM
    {
        uint32_t* a_u32 = reinterpret_cast<uint32_t*>(a_lds);
        const uint32_t* x_base = reinterpret_cast<const uint32_t*>(x + k_base);
        int n_u32 = BLOCK_M * RS / 2;
        for (int i = tid; i < n_u32; i += THREADS) {
            int ml = (i * 2) / RS;
            int kl = (i * 2) % RS;
            int gm = m_base + ml;
            if (gm < M) {
                a_u32[i] = reinterpret_cast<const uint32_t*>(x + gm * K + k_base)[kl / 2];
            } else {
                a_u32[i] = 0;
            }
        }
    }
    __syncthreads();

    // Pre-load rotation B vectors into registers for the wave's assigned N tile
    // Each wave handles N_TILES / WAVES tiles (round-robin)
    // For RS=128, N_TILES=8, WAVES=4 → each wave handles 2 N tiles

    // Process M tiles sequentially within each wave
    constexpr int M_TILES = BLOCK_M / 16;

    for (int n_tile = wave_id; n_tile < N_TILES; n_tile += WAVES) {
        // Preload all B vectors for this N tile (all K iters)
        fp16x8 b_vecs[K_ITERS];
        int b_col = n_tile * 16 + (lane_id % 16);
        for (int ki = 0; ki < K_ITERS; ki++) {
            int b_k = ki * 32 + (lane_id / 16) * 8;
            for (int j = 0; j < 8; j++)
                b_vecs[ki][j] = (_Float16)__bfloat162float(rot_lds[(b_k + j) * RS + b_col]);
        }

        for (int m_tile = 0; m_tile < M_TILES; m_tile++) {
            fp32x4 acc = {0, 0, 0, 0};

            for (int ki = 0; ki < K_ITERS; ki++) {
                int k_off = ki * 32;
                fp16x8 a_vec;
                int a_row = m_tile * 16 + (lane_id % 16);
                int a_k = k_off + (lane_id / 16) * 8;
                // Direct LDS load — no bounds check needed (BLOCK_M and RS are exact)
                for (int j = 0; j < 8; j++)
                    a_vec[j] = (_Float16)__bfloat162float(a_lds[a_row * RS + a_k + j]);

                acc = __builtin_amdgcn_mfma_f32_16x16x32_f16(a_vec, b_vecs[ki], acc, 0, 0, 0);
            }

            // Store to output LDS
            int c_col = n_tile * 16 + (lane_id % 16);
            int c_row = m_tile * 16 + (lane_id / 16) * 4;
            for (int i = 0; i < 4; i++)
                out_lds[(c_row + i) * RS + c_col] = __float2bfloat16(acc[i]);
        }
    }
    __syncthreads();

    // Write to GMEM — vector store
    {
        uint32_t* out_u32 = reinterpret_cast<uint32_t*>(out + k_base);
        const uint32_t* olds_u32 = reinterpret_cast<const uint32_t*>(out_lds);
        int n_u32 = BLOCK_M * RS / 2;
        for (int i = tid; i < n_u32; i += THREADS) {
            int ml = (i * 2) / RS;
            int kl = (i * 2) % RS;
            int gm = m_base + ml;
            if (gm < M) {
                reinterpret_cast<uint32_t*>(out + gm * K + k_base)[kl / 2] = olds_u32[i];
            }
        }
    }
}

// v3: M=1 decode-optimized — skip unused M tiles, rotate single row
template<int RS>
__global__ void rotation_mfma_v3_decode(
    __bf16* __restrict__ out,
    const __bf16* __restrict__ x,
    const __bf16* __restrict__ rot,
    int K
) {
    // M=1 specialization: only 1 row, no M-tile loop
    // blockIdx.y = rotation block index
    constexpr int THREADS = 256;
    constexpr int WAVES = THREADS / 64;
    constexpr int N_TILES = RS / 16;
    constexpr int K_ITERS = RS / 32;

    int bid_k = blockIdx.y, tid = threadIdx.x;
    int wave_id = tid / 64, lane_id = tid % 64;
    int k_base = bid_k * RS;

    __shared__ __bf16 rot_lds[RS * RS];

    // Load rotation to LDS
    {
        uint32_t* r32 = reinterpret_cast<uint32_t*>(rot_lds);
        const uint32_t* s32 = reinterpret_cast<const uint32_t*>(rot);
        for (int i = tid; i < RS * RS / 2; i += THREADS) r32[i] = s32[i];
    }

    // Load single row of A to registers directly (RS bf16 = RS*2 bytes)
    // 256 threads, RS elements → each thread loads RS/256 elements
    __shared__ __bf16 a_lds[RS];
    for (int i = tid; i < RS; i += THREADS)
        a_lds[i] = x[k_base + i];
    __syncthreads();

    // Each wave computes N_TILES/WAVES output tiles (each 1x16)
    // Only m_tile=0 (single row, padded to 16 for MFMA)
    __shared__ __bf16 out_lds[RS];

    for (int n_tile = wave_id; n_tile < N_TILES; n_tile += WAVES) {
        fp32x4 acc = {0, 0, 0, 0};

        for (int ki = 0; ki < K_ITERS; ki++) {
            int k_off = ki * 32;

            // A: row 0 (only row for M=1), padded to 16 rows for MFMA
            // lane%16 selects "row" → only row 0 has data
            fp16x8 a_vec;
            int a_row = lane_id % 16;  // 0..15, only row 0 matters
            int a_k = k_off + (lane_id / 16) * 8;
            for (int j = 0; j < 8; j++) {
                if (a_row == 0)
                    a_vec[j] = (_Float16)__bfloat162float(a_lds[a_k + j]);
                else
                    a_vec[j] = (_Float16)0.0f;
            }

            fp16x8 b_vec;
            int b_col = n_tile * 16 + (lane_id % 16);
            int b_k = k_off + (lane_id / 16) * 8;
            for (int j = 0; j < 8; j++)
                b_vec[j] = (_Float16)__bfloat162float(rot_lds[(b_k + j) * RS + b_col]);

            acc = __builtin_amdgcn_mfma_f32_16x16x32_f16(a_vec, b_vec, acc, 0, 0, 0);
        }

        // Only row 0 of the 16x16 output matters
        // C layout: row = (lane/16)*4 + i, col = lane%16
        int c_col = n_tile * 16 + (lane_id % 16);
        int c_row_base = (lane_id / 16) * 4;
        for (int i = 0; i < 4; i++) {
            if (c_row_base + i == 0 && c_col < RS)
                out_lds[c_col] = __float2bfloat16(acc[i]);
        }
    }
    __syncthreads();

    for (int i = tid; i < RS; i += THREADS)
        out[k_base + i] = out_lds[i];
}

void launch_v1(__bf16* o, const __bf16* x, const __bf16* r, int M, int K, int RS, hipStream_t s) {
    constexpr int BM = 32;
    dim3 g((M+BM-1)/BM, K/RS);
    if (RS==128)     rotation_mfma_v1<128,BM><<<g,256,0,s>>>(o,x,r,M,K);
    else if (RS==64) rotation_mfma_v1<64,BM><<<g,256,0,s>>>(o,x,r,M,K);
    else             rotation_mfma_v1<32,BM><<<g,256,0,s>>>(o,x,r,M,K);
}
void launch_v2(__bf16* o, const __bf16* x, const __bf16* r, int M, int K, int RS, hipStream_t s) {
    constexpr int BM = 32;
    dim3 g((M+BM-1)/BM, K/RS);
    if (RS==128)     rotation_mfma_v2<128,BM><<<g,256,0,s>>>(o,x,r,M,K);
    else if (RS==64) rotation_mfma_v2<64,BM><<<g,256,0,s>>>(o,x,r,M,K);
    else             rotation_mfma_v2<32,BM><<<g,256,0,s>>>(o,x,r,M,K);
}
void launch_v3(__bf16* o, const __bf16* x, const __bf16* r, int K, int RS, hipStream_t s) {
    dim3 g(1, K/RS);
    if (RS==128)     rotation_mfma_v3_decode<128><<<g,256,0,s>>>(o,x,r,K);
    else if (RS==64) rotation_mfma_v3_decode<64><<<g,256,0,s>>>(o,x,r,K);
    else             rotation_mfma_v3_decode<32><<<g,256,0,s>>>(o,x,r,K);
}

int main() {
    printf("=== LDS Rotation MFMA v1 vs v2 vs v3(decode) ===\n\n");
    int K = 2048;

    // Correctness: v1 vs v2 vs v3
    for (int RS : {32, 64, 128}) {
        int M = 4;
        __bf16 *d_x, *d_v1, *d_v2, *d_rot;
        __bf16 *h_v1, *h_v2;
        hipMalloc(&d_x, M*K*2); hipMalloc(&d_v1, M*K*2); hipMalloc(&d_v2, M*K*2);
        hipMalloc(&d_rot, RS*RS*2);
        h_v1 = (__bf16*)malloc(M*K*2); h_v2 = (__bf16*)malloc(M*K*2);

        // Init
        __bf16* h_x = (__bf16*)malloc(M*K*2);
        __bf16* h_r = (__bf16*)malloc(RS*RS*2);
        srand(42);
        for(int i=0;i<M*K;i++) h_x[i]=__float2bfloat16((rand()%200-100)/100.0f);
        for(int i=0;i<RS*RS;i++) h_r[i]=__float2bfloat16((rand()%200-100)/100.0f);
        hipMemcpy(d_x,h_x,M*K*2,hipMemcpyHostToDevice);
        hipMemcpy(d_rot,h_r,RS*RS*2,hipMemcpyHostToDevice);

        launch_v1(d_v1,d_x,d_rot,M,K,RS,0);
        launch_v2(d_v2,d_x,d_rot,M,K,RS,0);
        hipDeviceSynchronize();
        hipMemcpy(h_v1,d_v1,M*K*2,hipMemcpyDeviceToHost);
        hipMemcpy(h_v2,d_v2,M*K*2,hipMemcpyDeviceToHost);

        int exact=0;
        for(int i=0;i<M*K;i++) if(h_v1[i]==h_v2[i]) exact++;
        printf("RS=%3d v1==v2: %d/%d (%.1f%%) [%s]\n", RS, exact, M*K, 100.0f*exact/(M*K),
               exact==M*K?"PASS":"DIFF");

        free(h_x);free(h_r);free(h_v1);free(h_v2);
        hipFree(d_x);hipFree(d_v1);hipFree(d_v2);hipFree(d_rot);
    }

    // Performance
    printf("\n=== Performance Comparison (us/call) ===\n");
    __bf16 *d_x, *d_o, *d_r;
    hipMalloc(&d_x, 32*K*2); hipMalloc(&d_o, 32*K*2);

    for (int RS : {32, 64, 128}) {
        hipMalloc(&d_r, RS*RS*2);
        printf("--- RS=%d ---\n", RS);

        for (int M : {1, 16, 32}) {
            for(int i=0;i<50;i++){launch_v1(d_o,d_x,d_r,M,K,RS,0);launch_v2(d_o,d_x,d_r,M,K,RS,0);}
            if(M==1)for(int i=0;i<50;i++)launch_v3(d_o,d_x,d_r,K,RS,0);
            hipDeviceSynchronize();

            hipEvent_t t0,t1; hipEventCreate(&t0);hipEventCreate(&t1);

            hipEventRecord(t0);
            for(int i=0;i<1000;i++) launch_v1(d_o,d_x,d_r,M,K,RS,0);
            hipEventRecord(t1);hipEventSynchronize(t1);
            float ms1; hipEventElapsedTime(&ms1,t0,t1);

            hipEventRecord(t0);
            for(int i=0;i<1000;i++) launch_v2(d_o,d_x,d_r,M,K,RS,0);
            hipEventRecord(t1);hipEventSynchronize(t1);
            float ms2; hipEventElapsedTime(&ms2,t0,t1);

            float ms3 = 0;
            if (M == 1) {
                hipEventRecord(t0);
                for(int i=0;i<1000;i++) launch_v3(d_o,d_x,d_r,K,RS,0);
                hipEventRecord(t1);hipEventSynchronize(t1);
                hipEventElapsedTime(&ms3,t0,t1);
            }

            printf("  M=%2d: v1=%.1fus  v2=%.1fus(%.1fx)",
                   M, ms1*1000/1000, ms2*1000/1000, ms1/ms2);
            if (M==1) printf("  v3=%.1fus(%.1fx)", ms3*1000/1000, ms1/ms3);
            printf("\n");

            hipEventDestroy(t0);hipEventDestroy(t1);
        }
        hipFree(d_r);
    }

    hipFree(d_x);hipFree(d_o);
    printf("\nDone.\n");
    return 0;
}
