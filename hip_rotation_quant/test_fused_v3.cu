// Fused Rotation+Quant MFMA v3
// Optimizations over v2:
// 1. Vector LDS load for A and B (load 4 bf16 = 8 bytes at once via uint64)
// 2. Unrolled inner loops with #pragma unroll
// 3. Template everything for compile-time optimization

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using fp16x8 = __attribute__((ext_vector_type(8))) _Float16;
using fp32x4 = __attribute__((ext_vector_type(4))) float;

template<int RS, int BLOCK_M>
__global__ __launch_bounds__(256)
void fused_rot_quant_v3(
    uint8_t* __restrict__ out_fp4, uint8_t* __restrict__ out_scale,
    const __bf16* __restrict__ x, const __bf16* __restrict__ rot,
    int M, int K
) {
    constexpr int THREADS = 256;
    constexpr int WAVES = 4;
    constexpr int N_TILES = RS / 16;
    constexpr int K_ITERS = RS / 32;
    constexpr int M_TILES = BLOCK_M / 16;
    constexpr int QG = 32;
    constexpr int NUM_QG = RS / QG;
    constexpr int TOTAL_OUT_TILES = M_TILES * N_TILES;

    const int bid_m = blockIdx.x, bid_k = blockIdx.y, tid = threadIdx.x;
    const int wave_id = tid / 64, lane_id = tid % 64;
    const int m_base = bid_m * BLOCK_M, k_base = bid_k * RS;

    __shared__ __bf16 rot_lds[RS * RS];
    __shared__ __bf16 a_lds[BLOCK_M * RS];
    __shared__ float rotated[BLOCK_M * RS];

    // === Load rotation & A via 64-bit vector loads ===
    {
        uint64_t* dst = reinterpret_cast<uint64_t*>(rot_lds);
        const uint64_t* src = reinterpret_cast<const uint64_t*>(rot);
        constexpr int n64 = RS * RS / 4;  // 4 bf16 per uint64
        #pragma unroll 4
        for (int i = tid; i < n64; i += THREADS) dst[i] = src[i];
    }
    {
        constexpr int n64 = BLOCK_M * RS / 4;
        #pragma unroll 4
        for (int i = tid; i < n64; i += THREADS) {
            int elem = i * 4;
            int ml = elem / RS, kl = elem % RS;
            int gm = m_base + ml;
            if (gm < M) {
                reinterpret_cast<uint64_t*>(a_lds)[i] =
                    *reinterpret_cast<const uint64_t*>(&x[gm * K + k_base + kl]);
            } else {
                reinterpret_cast<uint64_t*>(a_lds)[i] = 0ULL;
            }
        }
    }
    __syncthreads();

    // === MFMA Rotation with preloaded B vectors ===
    for (int n_tile = wave_id; n_tile < N_TILES; n_tile += WAVES) {
        fp16x8 b_vecs[K_ITERS];
        const int b_col = n_tile * 16 + (lane_id % 16);
        const int b_k_group = (lane_id / 16) * 8;

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            const int b_k_base = ki * 32 + b_k_group;
            #pragma unroll
            for (int j = 0; j < 8; j++)
                b_vecs[ki][j] = (_Float16)__bfloat162float(rot_lds[(b_k_base + j) * RS + b_col]);
        }

        #pragma unroll
        for (int m_tile = 0; m_tile < M_TILES; m_tile++) {
            fp32x4 acc = {0, 0, 0, 0};
            const int a_row = m_tile * 16 + (lane_id % 16);
            const int a_k_group = (lane_id / 16) * 8;

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++) {
                fp16x8 a_vec;
                const int a_k_base = ki * 32 + a_k_group;
                const __bf16* a_ptr = &a_lds[a_row * RS + a_k_base];
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    a_vec[j] = (_Float16)__bfloat162float(a_ptr[j]);

                acc = __builtin_amdgcn_mfma_f32_16x16x32_f16(a_vec, b_vecs[ki], acc, 0, 0, 0);
            }

            const int c_col = n_tile * 16 + (lane_id % 16);
            const int c_row = m_tile * 16 + (lane_id / 16) * 4;
            #pragma unroll
            for (int i = 0; i < 4; i++)
                rotated[(c_row + i) * RS + c_col] = acc[i];
        }
    }
    __syncthreads();

    // === Quantization ===
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

    for (int i = tid; i < BLOCK_M * NUM_QG; i += THREADS) {
        int ml = i / NUM_QG, qg = i % NUM_QG;
        float amax = __int_as_float(*reinterpret_cast<int*>(&grp_amax[ml][qg]));
        uint32_t au = __float_as_uint(amax), rd = (au + 0x200000u) & 0xFF800000u;
        float su = floorf(log2f(__uint_as_float(rd))) - 2.0f;
        su = fmaxf(fminf(su, 127.0f), -127.0f);
        grp_e8m0[ml][qg] = (uint8_t)((int)su + 127);
        grp_scale[ml][qg] = exp2f(su);
    }
    __syncthreads();

    #pragma unroll 4
    for (int i = tid; i < BLOCK_M * (RS / 2); i += THREADS) {
        int ml = i / (RS / 2), pi = i % (RS / 2), kl = pi * 2, gm = m_base + ml;
        if (gm >= M) continue;
        float v0 = rotated[ml * RS + kl], v1 = rotated[ml * RS + kl + 1];
        float sc = grp_scale[ml][kl / QG];
        uint32_t packed;
        asm volatile("v_cvt_scalef32_pk_fp4_f32 %0,%1,%2,%3" : "=v"(packed) : "v"(v0), "v"(v1), "v"(sc));
        out_fp4[gm * (K / 2) + k_base / 2 + pi] = (uint8_t)packed;
    }

    for (int i = tid; i < BLOCK_M * NUM_QG; i += THREADS) {
        int ml = i / NUM_QG, qg = i % NUM_QG, gm = m_base + ml;
        if (gm >= M) return;
        out_scale[gm * (K / QG) + k_base / QG + qg] = grp_e8m0[ml][qg];
    }
}

// Previous v2 kernel for comparison (same as test_fused_rot_quant_mfma.cu)
template<int RS, int BLOCK_M>
__global__ void fused_rot_quant_v2(
    uint8_t* __restrict__ out_fp4, uint8_t* __restrict__ out_scale,
    const __bf16* __restrict__ x, const __bf16* __restrict__ rot,
    int M, int K
) {
    constexpr int THREADS=256,WAVES=4,N_TILES=RS/16,K_ITERS=RS/32,M_TILES=BLOCK_M/16;
    constexpr int QG=32,NUM_QG=RS/QG;
    int bid_m=blockIdx.x,bid_k=blockIdx.y,tid=threadIdx.x;
    int wave_id=tid/64,lane_id=tid%64;
    int m_base=bid_m*BLOCK_M,k_base=bid_k*RS;
    __shared__ __bf16 rot_lds[RS*RS]; __shared__ __bf16 a_lds[BLOCK_M*RS]; __shared__ float rotated[BLOCK_M*RS];
    {uint32_t*r=(uint32_t*)rot_lds;const uint32_t*s=(const uint32_t*)rot;
     for(int i=tid;i<RS*RS/2;i+=THREADS)r[i]=s[i];}
    for(int i=tid;i<BLOCK_M*RS;i+=THREADS){int ml=i/RS,kl=i%RS,gm=m_base+ml;
        if(gm<M)a_lds[i]=x[gm*K+k_base+kl];else a_lds[i]=__float2bfloat16(0.f);}
    __syncthreads();
    for(int nt=wave_id;nt<N_TILES;nt+=WAVES){
        fp16x8 bv[K_ITERS];int bc=nt*16+(lane_id%16);
        for(int ki=0;ki<K_ITERS;ki++){int bk=ki*32+(lane_id/16)*8;
            for(int j=0;j<8;j++)bv[ki][j]=(_Float16)__bfloat162float(rot_lds[(bk+j)*RS+bc]);}
        for(int mt=0;mt<M_TILES;mt++){
            fp32x4 acc={0,0,0,0};int ar=mt*16+(lane_id%16),akg=(lane_id/16)*8;
            for(int ki=0;ki<K_ITERS;ki++){fp16x8 av;int ak=ki*32+akg;
                for(int j=0;j<8;j++)av[j]=(_Float16)__bfloat162float(a_lds[ar*RS+ak+j]);
                acc=__builtin_amdgcn_mfma_f32_16x16x32_f16(av,bv[ki],acc,0,0,0);}
            int cc=nt*16+(lane_id%16),cr=mt*16+(lane_id/16)*4;
            for(int i=0;i<4;i++)rotated[(cr+i)*RS+cc]=acc[i];}}
    __syncthreads();
    __shared__ float ga[BLOCK_M][NUM_QG];__shared__ uint8_t ge[BLOCK_M][NUM_QG];__shared__ float gs[BLOCK_M][NUM_QG];
    for(int i=tid;i<BLOCK_M*NUM_QG;i+=THREADS)ga[i/NUM_QG][i%NUM_QG]=0.f;
    __syncthreads();
    for(int i=tid;i<BLOCK_M*RS;i+=THREADS){int ml=i/RS,kl=i%RS,gm=m_base+ml;if(gm>=M)continue;
        atomicMax((int*)&ga[ml][kl/QG],__float_as_int(fabsf(rotated[i])));}
    __syncthreads();
    for(int i=tid;i<BLOCK_M*NUM_QG;i+=THREADS){int ml=i/NUM_QG,qg=i%NUM_QG;
        float am=__int_as_float(*(int*)&ga[ml][qg]);uint32_t au=__float_as_uint(am),rd=(au+0x200000u)&0xFF800000u;
        float su=floorf(log2f(__uint_as_float(rd)))-2.f;su=fmaxf(fminf(su,127.f),-127.f);
        ge[ml][qg]=(uint8_t)((int)su+127);gs[ml][qg]=exp2f(su);}
    __syncthreads();
    for(int i=tid;i<BLOCK_M*(RS/2);i+=THREADS){int ml=i/(RS/2),pi=i%(RS/2),kl=pi*2,gm=m_base+ml;
        if(gm>=M)continue;float v0=rotated[ml*RS+kl],v1=rotated[ml*RS+kl+1],sc=gs[ml][kl/QG];
        uint32_t pk;asm volatile("v_cvt_scalef32_pk_fp4_f32 %0,%1,%2,%3":"=v"(pk):"v"(v0),"v"(v1),"v"(sc));
        out_fp4[gm*(K/2)+k_base/2+pi]=(uint8_t)pk;}
    for(int i=tid;i<BLOCK_M*NUM_QG;i+=THREADS){int ml=i/NUM_QG,qg=i%NUM_QG,gm=m_base+ml;if(gm>=M)return;
        out_scale[gm*(K/QG)+k_base/QG+qg]=ge[ml][qg];}
}

template<int RS>
void launch_v2(uint8_t*f,uint8_t*s,const __bf16*x,const __bf16*r,int M,int K,hipStream_t st){
    constexpr int BM=32;dim3 g((M+BM-1)/BM,K/RS);
    fused_rot_quant_v2<RS,BM><<<g,256,0,st>>>(f,s,x,r,M,K);}
template<int RS>
void launch_v3(uint8_t*f,uint8_t*s,const __bf16*x,const __bf16*r,int M,int K,hipStream_t st){
    constexpr int BM=32;dim3 g((M+BM-1)/BM,K/RS);
    fused_rot_quant_v3<RS,BM><<<g,256,0,st>>>(f,s,x,r,M,K);}

int main(){
    printf("=== Fused Rot+Quant: v2 vs v3 (template optimized) ===\n\n");
    int K=2048;

    // Correctness
    for(int RS:{32,64,128}){
        int M=64;
        __bf16*h_x=(__bf16*)malloc(M*K*2),*h_r=(__bf16*)malloc(RS*RS*2);
        srand(42);for(int i=0;i<M*K;i++)h_x[i]=__float2bfloat16((rand()%1000-500)/500.f);
        for(int i=0;i<RS*RS;i++)h_r[i]=__float2bfloat16((rand()%1000-500)/500.f);
        __bf16*d_x,*d_r;uint8_t*d_f2,*d_s2,*d_f3,*d_s3;
        hipMalloc(&d_x,M*K*2);hipMalloc(&d_r,RS*RS*2);
        hipMalloc(&d_f2,M*(K/2));hipMalloc(&d_s2,M*(K/32));
        hipMalloc(&d_f3,M*(K/2));hipMalloc(&d_s3,M*(K/32));
        hipMemcpy(d_x,h_x,M*K*2,hipMemcpyHostToDevice);hipMemcpy(d_r,h_r,RS*RS*2,hipMemcpyHostToDevice);
        if(RS==128){launch_v2<128>(d_f2,d_s2,(const __bf16*)d_x,(const __bf16*)d_r,M,K,0);
                    launch_v3<128>(d_f3,d_s3,(const __bf16*)d_x,(const __bf16*)d_r,M,K,0);}
        else if(RS==64){launch_v2<64>(d_f2,d_s2,(const __bf16*)d_x,(const __bf16*)d_r,M,K,0);
                        launch_v3<64>(d_f3,d_s3,(const __bf16*)d_x,(const __bf16*)d_r,M,K,0);}
        else{launch_v2<32>(d_f2,d_s2,(const __bf16*)d_x,(const __bf16*)d_r,M,K,0);
             launch_v3<32>(d_f3,d_s3,(const __bf16*)d_x,(const __bf16*)d_r,M,K,0);}
        hipDeviceSynchronize();
        uint8_t*hf2=(uint8_t*)malloc(M*(K/2)),*hs2=(uint8_t*)malloc(M*(K/32));
        uint8_t*hf3=(uint8_t*)malloc(M*(K/2)),*hs3=(uint8_t*)malloc(M*(K/32));
        hipMemcpy(hf2,d_f2,M*(K/2),hipMemcpyDeviceToHost);hipMemcpy(hs2,d_s2,M*(K/32),hipMemcpyDeviceToHost);
        hipMemcpy(hf3,d_f3,M*(K/2),hipMemcpyDeviceToHost);hipMemcpy(hs3,d_s3,M*(K/32),hipMemcpyDeviceToHost);
        int sm=0,fm=0;
        for(int i=0;i<M*(K/32);i++)if(hs2[i]==hs3[i])sm++;
        for(int i=0;i<M*(K/2);i++)if(hf2[i]==hf3[i])fm++;
        printf("RS=%3d: v2==v3 scale=%d/%d(%.0f%%) fp4=%d/%d(%.0f%%) [%s]\n",
               RS,sm,M*(K/32),100.f*sm/(M*(K/32)),fm,M*(K/2),100.f*fm/(M*(K/2)),
               (sm==M*(K/32)&&fm==M*(K/2))?"PASS":"DIFF");
        free(h_x);free(h_r);free(hf2);free(hs2);free(hf3);free(hs3);
        hipFree(d_x);hipFree(d_r);hipFree(d_f2);hipFree(d_s2);hipFree(d_f3);hipFree(d_s3);
    }

    // Performance
    printf("\n=== Performance v2 vs v3 (us/call) ===\n");
    __bf16*d_x,*d_r;uint8_t*d_f,*d_s;
    hipMalloc(&d_x,2048*K*2);hipMalloc(&d_f,2048*(K/2));hipMalloc(&d_s,2048*(K/32));

    for(int RS:{32,64,128}){
        hipMalloc(&d_r,RS*RS*2);
        printf("--- RS=%d ---\n",RS);
        for(int M:{1,4,16,32,64,128,256,512,1024}){
            for(int i=0;i<20;i++){
                if(RS==128){launch_v2<128>(d_f,d_s,(const __bf16*)d_x,(const __bf16*)d_r,M,K,0);
                            launch_v3<128>(d_f,d_s,(const __bf16*)d_x,(const __bf16*)d_r,M,K,0);}
                else if(RS==64){launch_v2<64>(d_f,d_s,(const __bf16*)d_x,(const __bf16*)d_r,M,K,0);
                                launch_v3<64>(d_f,d_s,(const __bf16*)d_x,(const __bf16*)d_r,M,K,0);}
                else{launch_v2<32>(d_f,d_s,(const __bf16*)d_x,(const __bf16*)d_r,M,K,0);
                     launch_v3<32>(d_f,d_s,(const __bf16*)d_x,(const __bf16*)d_r,M,K,0);}}
            hipDeviceSynchronize();
            hipEvent_t t0,t1;hipEventCreate(&t0);hipEventCreate(&t1);int NI=500;
            hipEventRecord(t0);
            for(int i=0;i<NI;i++){
                if(RS==128)launch_v2<128>(d_f,d_s,(const __bf16*)d_x,(const __bf16*)d_r,M,K,0);
                else if(RS==64)launch_v2<64>(d_f,d_s,(const __bf16*)d_x,(const __bf16*)d_r,M,K,0);
                else launch_v2<32>(d_f,d_s,(const __bf16*)d_x,(const __bf16*)d_r,M,K,0);}
            hipEventRecord(t1);hipEventSynchronize(t1);float ms2;hipEventElapsedTime(&ms2,t0,t1);
            hipEventRecord(t0);
            for(int i=0;i<NI;i++){
                if(RS==128)launch_v3<128>(d_f,d_s,(const __bf16*)d_x,(const __bf16*)d_r,M,K,0);
                else if(RS==64)launch_v3<64>(d_f,d_s,(const __bf16*)d_x,(const __bf16*)d_r,M,K,0);
                else launch_v3<32>(d_f,d_s,(const __bf16*)d_x,(const __bf16*)d_r,M,K,0);}
            hipEventRecord(t1);hipEventSynchronize(t1);float ms3;hipEventElapsedTime(&ms3,t0,t1);
            printf("  M=%4d: v2=%.1fus v3=%.1fus (%.2fx)\n",M,ms2*1000/NI,ms3*1000/NI,ms2/ms3);
            hipEventDestroy(t0);hipEventDestroy(t1);}
        hipFree(d_r);
    }
    hipFree(d_x);hipFree(d_f);hipFree(d_s);
    printf("\nDone.\n");return 0;
}
