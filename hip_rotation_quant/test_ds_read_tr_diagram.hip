// Verify DS_READ_B64_TR_B16 layout exactly as shown in AMD diagram:
//
// LDS (4x16 bf16, 128 bytes contiguous):
//        col: 0  1  2  3 | 4  5  6  7 | 8  9 10 11 | 12 13 14 15
//  row 0:    t0 (4 bf16) | t1 (4 bf16)| t2 (4 bf16)| t3 (4 bf16)
//  row 1:    t4          | t5         | t6         | t7
//  row 2:    t8          | t9         | t10        | t11
//  row 3:    t12         | t13        | t14        | t15
//
// After transpose → VGPR (16x4):
//  t0  gets: [row0_col0, row1_col0, row2_col0, row3_col0]
//  t1  gets: [row0_col1, row1_col1, row2_col1, row3_col1]
//  ...
//  t15 gets: [row0_col15, row1_col15, row2_col15, row3_col15]

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

using bf16x4_t = __attribute__((ext_vector_type(4))) __bf16;

__global__ __attribute__((amdgpu_flat_work_group_size(64, 64)))
void test_diagram_layout(float* __restrict__ out)
{
    __shared__ __bf16 lds[4][16]; // exactly 4×16 = 128 bytes contiguous

    int lane_id = threadIdx.x;

    // Fill LDS with unique values: lds[r][c] = r * 16 + c  (0..63)
    if (lane_id < 64) {
        int r = lane_id / 16, c = lane_id % 16;
        if (r < 4) lds[r][c] = __float2bfloat16((float)(r * 16 + c));
    }
    __syncthreads();

    if (lane_id >= 16) return;

    // Per-lane address matching the diagram:
    // t_i sits at LDS row (i/4), col_start (i%4)*4
    // addr = &lds[i/4][(i%4)*4]
    int my_row = lane_id / 4;
    int my_col_start = (lane_id % 4) * 4;

    auto* ptr = reinterpret_cast<__attribute__((address_space(3))) bf16x4_t*>(
        (uintptr_t)&lds[my_row][my_col_start]);
    bf16x4_t result = __builtin_amdgcn_ds_read_tr16_b64_v4bf16(ptr);

    // Store ds_read_tr result
    for (int i = 0; i < 4; i++)
        out[lane_id * 8 + i] = __bfloat162float(result[i]);

    // Store expected: col lane_id from rows 0-3
    for (int r = 0; r < 4; r++)
        out[lane_id * 8 + 4 + r] = __bfloat162float(lds[r][lane_id]);
}

extern "C" void launch_test(float* out, hipStream_t s) {
    test_diagram_layout<<<1, 64, 0, s>>>(out);
}
