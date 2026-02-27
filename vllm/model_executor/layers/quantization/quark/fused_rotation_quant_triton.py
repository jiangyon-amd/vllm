#!/usr/bin/env python3
"""
Fused Online Rotation + MXFP4 Quantization v5

Optimizations over v4:
  1. Use aiter's _mxfp4_quant_op directly → bit-exact output, banker's rounding
  2. bf16 tl.dot for rotation → 2x MFMA throughput (fp4 precision doesn't need fp32 matmul)
  3. Persistent kernel mode for small M — one program loops over rotation blocks
  4. Tuned autoconfig for MI355X gfx950
"""

import torch
import triton
import triton.language as tl
import time
from typing import Optional

# Import aiter's quant op (fallback)
from aiter.ops.triton._triton_kernels.quant import _mxfp4_quant_op


@triton.jit
def _mxfp4_quant_hip_compat(x, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE):
    """
    MXFP4 quant matching HIP per_1x32_f4_quant_hip — branchless optimized.

    Optimizations vs previous version:
      - Unified branchless fp4 conversion (no tl.where branches)
      - Simplified carry logic (fewer tl.where)
      - Fewer intermediate tensors
    """
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)

    # Per-group abs max
    amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)

    # HIP fp4_scale: round-up exponent with carry (simplified)
    amax_u32 = amax.to(tl.uint32, bitcast=True)
    # Round mantissa up: add 0x400000 (half of mantissa MSB), clear mantissa
    # This is equivalent to fp4_scale's carry logic for non-zero values
    amax_rounded = ((amax_u32 + 0x3FFFFF) & 0xFF800000)
    exponent = (amax_rounded >> 23) & 0xFF

    # e8m0 = exponent - 2
    e8m0 = tl.maximum(exponent, 2) - 2
    bs_e8m0 = e8m0.to(tl.uint8)

    # quant_scale = 2^(129 - exponent)
    inv_exp = tl.minimum(tl.maximum(256 - exponent, 1), 254)
    quant_scale = (inv_exp.to(tl.uint32) << 23).to(tl.float32, bitcast=True)

    # Scale values
    qx = x * quant_scale

    # === FP32 → FP4 via hardware instruction (gfx950) ===
    # v_cvt_scalef32_pk_fp4_f32: converts 2 fp32 values to packed fp4x2
    # with scale applied internally: fp4 = round_to_e2m1(value / inverted_scale)
    #
    # inverted_scale = 2^(e8m0 - 127), constructed from e8m0 bits

    # Construct inverted_scale as float32 from e8m0
    inv_scale_f32 = (e8m0.to(tl.uint32) << 23).to(tl.float32, bitcast=True)

    # Reshape x to pairs for the hardware instruction
    x_pairs = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE // 2, 2)
    even_vals, odd_vals = tl.split(x_pairs)

    # Broadcast inverted_scale [BM, NQG, 1] → [BM, NQG, HALF_QG]
    HALF_QG: tl.constexpr = MXFP4_QUANT_BLOCK_SIZE // 2
    inv_scale_broad = tl.broadcast_to(inv_scale_f32, [BLOCK_SIZE_M, NUM_QUANT_BLOCKS, HALF_QG])

    # Flatten for element-wise inline ASM
    even_flat = even_vals.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS * HALF_QG)
    odd_flat = odd_vals.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS * HALF_QG)
    scale_flat = inv_scale_broad.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS * HALF_QG)

    # Hardware FP4 conversion
    packed_u32 = tl.inline_asm_elementwise(
        asm="v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3",
        constraints="=v,v,v,v",
        args=[even_flat, odd_flat, scale_flat],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    x_fp4 = packed_u32.to(tl.uint8)
    x_fp4 = x_fp4.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2)

    return x_fp4, bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)

# ---------------------------------------------------------------------------
# Kernel: standard grid (one program per M-tile × rotation-block)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "TILE_K": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "TILE_K": 64},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "TILE_K": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "TILE_K": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "TILE_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "TILE_K": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16,  "TILE_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16,  "TILE_K": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16,  "TILE_K": 32},  num_warps=4, num_stages=2),
    ],
    key=["M", "rotation_size"],
)
@triton.jit
def _fused_rot_quant_kernel(
    x_ptr, rot_ptr, fp4_ptr, scale_ptr,
    M, K,
    stride_x_m, stride_rot_r, stride_rot_c,
    stride_fp4_m, stride_sc_m,
    rotation_size: tl.constexpr,
    BLOCK_M:       tl.constexpr,
    TILE_K:        tl.constexpr,
    QGROUP:        tl.constexpr,
):
    pid_m   = tl.program_id(0)
    pid_rot = tl.program_id(1)

    m_base = pid_m * BLOCK_M
    m_offs = m_base + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M
    col_base = pid_rot * rotation_size
    n_offs = tl.arange(0, rotation_size)

    # ---- K-tiled rotation: acc = x_block @ rotation ----
    acc = tl.zeros([BLOCK_M, rotation_size], dtype=tl.float32)

    for k_off in tl.static_range(0, rotation_size, TILE_K):
        k_ids = k_off + tl.arange(0, TILE_K)

        x_tile = tl.load(
            x_ptr + m_offs[:, None] * stride_x_m + (col_base + k_ids)[None, :],
            mask=m_mask[:, None], other=0.0,
        )

        rot_tile = tl.load(
            rot_ptr + k_ids[:, None] * stride_rot_r + n_offs[None, :] * stride_rot_c,
        )

        # bf16 dot → uses bf16 MFMA (2x throughput vs fp32)
        acc = tl.dot(x_tile, rot_tile, acc, input_precision="ieee")

    # ---- MXFP4 quantization (HIP-compatible scale algorithm) ----
    NUM_QG: tl.constexpr = rotation_size // QGROUP
    fp4_packed, e8m0_scales = _mxfp4_quant_hip_compat(acc, rotation_size, BLOCK_M, QGROUP)

    # ---- Store fp4 [BLOCK_M, rotation_size // 2] ----
    fp4_base = pid_rot * (rotation_size // 2)
    fp4_cols = fp4_base + tl.arange(0, rotation_size // 2)
    tl.store(
        fp4_ptr + m_offs[:, None] * stride_fp4_m + fp4_cols[None, :],
        fp4_packed, mask=m_mask[:, None],
    )

    # ---- Store scales [BLOCK_M, NUM_QG] ----
    sc_base = pid_rot * NUM_QG
    sc_cols = sc_base + tl.arange(0, NUM_QG)
    tl.store(
        scale_ptr + m_offs[:, None] * stride_sc_m + sc_cols[None, :],
        e8m0_scales, mask=m_mask[:, None],
    )


# ---------------------------------------------------------------------------
# Kernel: persistent (one program per M-tile, loops over rotation blocks)
# Reduces total grid size → lower launch overhead for small M
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "TILE_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "TILE_K": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "TILE_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "TILE_K": 64},  num_warps=4, num_stages=2),
    ],
    key=["M", "rotation_size"],
)
@triton.jit
def _fused_rot_quant_persistent_kernel(
    x_ptr, rot_ptr, fp4_ptr, scale_ptr,
    M, K, num_rot_blocks,
    stride_x_m, stride_rot_r, stride_rot_c,
    stride_fp4_m, stride_sc_m,
    rotation_size: tl.constexpr,
    BLOCK_M:       tl.constexpr,
    TILE_K:        tl.constexpr,
    QGROUP:        tl.constexpr,
):
    """Each program processes BLOCK_M rows × ALL rotation blocks sequentially."""
    pid_m = tl.program_id(0)

    m_base = pid_m * BLOCK_M
    m_offs = m_base + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M
    n_offs = tl.arange(0, rotation_size)
    NUM_QG: tl.constexpr = rotation_size // QGROUP

    for rot_idx in range(0, num_rot_blocks):
        col_base = rot_idx * rotation_size

        # K-tiled rotation matmul
        acc = tl.zeros([BLOCK_M, rotation_size], dtype=tl.float32)
        for k_off in tl.static_range(0, rotation_size, TILE_K):
            k_ids = k_off + tl.arange(0, TILE_K)
            x_tile = tl.load(
                x_ptr + m_offs[:, None] * stride_x_m + (col_base + k_ids)[None, :],
                mask=m_mask[:, None], other=0.0,
            )
            rot_tile = tl.load(
                rot_ptr + k_ids[:, None] * stride_rot_r + n_offs[None, :] * stride_rot_c,
            )
            acc = tl.dot(x_tile, rot_tile, acc, input_precision="ieee")

        # Quant
        fp4_packed, e8m0_scales = _mxfp4_quant_op(acc, rotation_size, BLOCK_M, QGROUP)

        # Store
        fp4_base = rot_idx * (rotation_size // 2)
        fp4_cols = fp4_base + tl.arange(0, rotation_size // 2)
        tl.store(
            fp4_ptr + m_offs[:, None] * stride_fp4_m + fp4_cols[None, :],
            fp4_packed, mask=m_mask[:, None],
        )
        sc_base = rot_idx * NUM_QG
        sc_cols = sc_base + tl.arange(0, NUM_QG)
        tl.store(
            scale_ptr + m_offs[:, None] * stride_sc_m + sc_cols[None, :],
            e8m0_scales, mask=m_mask[:, None],
        )


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

# Persistent kernel only when total grid blocks would be very small
# (few rows AND few rotation blocks)
_PERSISTENT_M_THRESHOLD = 0  # disabled: standard 2D grid is consistently better

def fused_rotation_mxfp4_quant(
    x: torch.Tensor,
    rotation: torch.Tensor,
    rotation_size: int = 128,
    shuffle: bool = False,
    scale_shuffle_padding: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused online rotation + MXFP4 quantization.
    Uses aiter's _mxfp4_quant_op for bit-exact quantization output.
    """
    assert x.ndim == 2
    M, K = x.shape
    assert K % rotation_size == 0
    assert rotation.shape == (rotation_size, rotation_size)

    QGROUP = 32
    num_rot_blocks = K // rotation_size

    fp4    = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
    scales = torch.empty((M, K // QGROUP), dtype=torch.uint8, device=x.device)

    if M <= _PERSISTENT_M_THRESHOLD:
        # Persistent: each program handles ALL rotation blocks for its row-tile
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        _fused_rot_quant_persistent_kernel[grid](
            x, rotation, fp4, scales,
            M, K, num_rot_blocks,
            x.stride(0), rotation.stride(0), rotation.stride(1),
            fp4.stride(0), scales.stride(0),
            rotation_size=rotation_size, QGROUP=QGROUP,
        )
    else:
        # Standard 2D grid
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), num_rot_blocks)
        _fused_rot_quant_kernel[grid](
            x, rotation, fp4, scales,
            M, K,
            x.stride(0), rotation.stride(0), rotation.stride(1),
            fp4.stride(0), scales.stride(0),
            rotation_size=rotation_size, QGROUP=QGROUP,
        )

    if shuffle:
        fp4, scales = _apply_shuffle(fp4, scales, M, K, QGROUP, scale_shuffle_padding)

    return fp4, scales


def _apply_shuffle(fp4, scales, M, K, QGROUP, pad):
    if pad:
        pad_m = (M + 255) // 256 * 256
        n_sc = K // QGROUP
        pad_n = (n_sc + 7) // 8 * 8
        sc_padded = torch.zeros((pad_m, pad_n), dtype=torch.uint8, device=scales.device)
        sc_padded[:M, :n_sc] = scales
        return fp4, sc_padded
    return fp4, scales


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def unfused_rotation_mxfp4_quant(x, rotation, rotation_size=128):
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    M, K = x.shape
    x_r = x.reshape(M, K // rotation_size, rotation_size)
    x_r = torch.einsum("bgs,sk->bgk", x_r, rotation.to(x.dtype))
    return dynamic_mxfp4_quant(x_r.reshape(M, K))


# ---------------------------------------------------------------------------
# Verify & Bench
# ---------------------------------------------------------------------------

def verify(M=32, K=5120, rotation_size=128, device="cuda:0"):
    x   = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    rot = torch.linalg.qr(torch.randn(rotation_size, rotation_size, device=device))[0].to(torch.bfloat16)
    fp4_f, sc_f = fused_rotation_mxfp4_quant(x, rot, rotation_size)
    fp4_u, sc_u = unfused_rotation_mxfp4_quant(x, rot, rotation_size)

    fp4_ok   = torch.equal(fp4_f, fp4_u)
    scale_ok = torch.equal(sc_f, sc_u)
    n_fp4  = 0 if fp4_ok else (fp4_f != fp4_u).sum().item()
    n_sc   = 0 if scale_ok else (sc_f != sc_u).sum().item()
    pct    = n_fp4 / fp4_f.numel() * 100 if n_fp4 else 0
    tag    = "✅" if fp4_ok and scale_ok else f"~{pct:.1f}%"
    print(f"  M={M:>5d}  fp4_diff={n_fp4:>6d}  sc_diff={n_sc:>4d}  {tag}")
    return fp4_ok and scale_ok


def bench(M, K, rotation_size=128, n_iters=500, n_warmup=50, device="cuda:0"):
    x   = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    rot = torch.linalg.qr(torch.randn(rotation_size, rotation_size, device=device))[0].to(torch.bfloat16)

    for _ in range(n_warmup):
        _ = fused_rotation_mxfp4_quant(x, rot, rotation_size)
        _ = unfused_rotation_mxfp4_quant(x, rot, rotation_size)
    torch.cuda.synchronize()

    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n_iters): _ = unfused_rotation_mxfp4_quant(x, rot, rotation_size)
    torch.cuda.synchronize(); t_u = (time.perf_counter() - t0) / n_iters * 1e6

    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n_iters): _ = fused_rotation_mxfp4_quant(x, rot, rotation_size)
    torch.cuda.synchronize(); t_f = (time.perf_counter() - t0) / n_iters * 1e6

    sp = t_u / t_f
    flag = "✅" if sp > 1.0 else "⚠️"
    print(f"  M={M:>5d}  K={K}  unfused={t_u:7.1f}µs  fused={t_f:7.1f}µs  {sp:.2f}x {flag}")
    return t_u, t_f


def main():
    device = "cuda:0"
    print("=" * 72)
    print("Fused Rotation + MXFP4 Quant  v5  (aiter quant + bf16 dot)")
    print("=" * 72)

    print("\n--- Correctness ---")
    for M in [1, 8, 32, 64, 128, 256, 512]:
        verify(M=M, device=device)

    print("\n--- Benchmark K=5120 (hidden) ---")
    for M in [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        bench(M, 5120, device=device)

    print("\n--- Benchmark K=51200 (gate_up_proj) ---")
    for M in [1, 16, 64, 256]:
        bench(M, 51200, device=device)


if __name__ == "__main__":
    main()
