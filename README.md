# MXFP4 Online Rotation — Fused Kernels for vLLM

This directory contains the production Gluon (Triton) and HIP MFMA fused kernels for
MXFP4 online R₁ rotation on AMD Instinct MI355X (gfx950).

Companion to the blog post:
**From Accuracy to Efficiency: Fused Kernels for MXFP4 Online Rotation on AMD Instinct MI355X**.

## What is being fused?

For MXFP4 with learned rotation, **R₂** is merged offline into v_proj/o_proj weights
(zero inference cost). **R₁** is applied **online** at every layer to inputs of
`qkv_proj` and `gate_up_proj` — it cannot be safely fused offline for MXFP4 because
absorbing it into RMSNorm degrades quantization accuracy. The kernels in this
directory fuse the online R₁ rotation matmul together with the subsequent FP4
quantization (and the MoE expert sort, for MoE layers) into a single kernel launch,
keeping intermediate values in registers.

## File layout (`vllm/model_executor/layers/quantization/quark/`)

### Dense (8B / 14B / 32B)
| File | Role |
|------|------|
| `fused_rotation_quant_gluon.py`        | **Gluon Dense** fused kernel (rotation matmul → fp32 acc → FP4 quant in registers) |
| `fused_rotation_quant_hip.py`          | HIP MFMA Dense fallback (alternative backend) |
| `fused_rotation_mxfp4_quant.py`        | Reference Triton impl + standalone correctness/benchmark script |
| `schemes/quark_ocp_mx.py`              | Dense quark scheme — dispatcher that picks Gluon vs HIP vs separated |

### MoE (30B-A3B)
| File | Role |
|------|------|
| `fused_rotation_mxfp4_quant_moe_gluon_v2.py` | Gluon MoE kernel — used for M=1 / small M |
| `fused_rotation_mxfp4_quant_moe_gluon_v4.py` | Gluon MoE kernel — **3-in-1 fused** (rotation + FP4 quant + expert sort) production version |
| `fused_rotation_mxfp4_quant_moe_hip.py`      | HIP MFMA MoE fallback |
| `fused_rotation_mxfp4_quant_moe_sort.py`     | MoE dispatcher — chooses v2 vs v4 vs HIP based on tile config |
| `gluon_rotation_only_v2.py` / `_v3.py`       | Rotation-only kernel (no quant) used by aiter monkey-patch path |

### aiter integration
| File | Role |
|------|------|
| `aiter_rotation_patch.py` | Monkey-patches `aiter.fused_moe.fused_moe_2stages` to replace `torch.matmul` with a Gluon rotation kernel |
| `aiter_v4_patch.py`       | Monkey-patches aiter to use the 3-in-1 Gluon V4 fused kernel |
| `fused_moe_rotation.py`   | Thin re-export shim used by `qwen3_moe.py` to select the patch |

### Shared
| File | Role |
|------|------|
| `transform.py` | Detects which Linear layers should carry `input_rotation` (R₁), based on Quark's `scaling_layers.next_modules` |

## Runtime environment

| Variable | Effect |
|----------|--------|
| `VLLM_USE_FUSED_ROTATION_QUANT=1` | Enable Dense Gluon fused rotation+quant |
| `VLLM_MOE_FUSED_ROTATION=1`       | Enable MoE Gluon fused rotation+quant+sort |
| `VLLM_ROCM_USE_AITER=1`           | Enable aiter MoE backend (required for MoE) |
| `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1` | Enable aiter FP4 ASM GEMM |
| `VLLM_HIP_FUSED_MAX_M=N`          | Force HIP MFMA Dense for M<N (default 0 = always Gluon) |

## How it integrates with stock vLLM

These files drop into the existing `vllm/model_executor/layers/quantization/quark/`
package and replace / add to the upstream scheme. The scheme detects whether the
loaded model has rotation parameters (via Quark's exported `rotations.safetensors`)
and, if so, registers `input_rotation` on the relevant Linear layers. The dispatcher
in `schemes/quark_ocp_mx.py` then picks the fused kernel at apply time.

## Reproducing the blog results

See the blog post for the full methodology. Required versions:

* `rocm/vllm-dev:nightly_main_20260121` (or compatible)
* `amd-quark==0.11` with a model exported from
  [`Quark/examples/torch/language_modeling/rotation`](https://github.com/amd/Quark/tree/release/0.11/examples/torch/language_modeling/rotation)
  using `qwen3_train_r1_128_online_r2.json` (note: `online_r1_rotation: true`)
* AMD MI355X (gfx950)

## License

This code is licensed under the Apache License 2.0, the same as upstream vLLM.
Copyright © 2026 Advanced Micro Devices, Inc.
