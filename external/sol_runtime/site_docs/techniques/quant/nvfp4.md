# NVFP4

**NVFP4** is NVIDIA's 4-bit floating-point format (Blackwell) with per-block
scaling. Here the GEN/video linears are swapped to TransformerEngine `te.Linear`
with `NVFP4BlockScaling`: BF16 weights are copied in and quantized per-forward, and
the swap is **step-selective** — the first/last few denoise steps stay BF16 (most
quality-sensitive), the middle steps run FP4. On real Cosmos3-Super shapes the FP4
GEMMs are ~2.7–3.6× faster than BF16.

**In this repo.** Cosmos3 (`SGLANG_COSMOS3_FP4_LINEAR=1`, `FP4_TARGETS=gate_up,down,qkv,out`,
`FP4_SKIP_FIRST_STEPS` / `_LAST_STEPS`) and LTX video FFN (`SGLANG_HQ_ENABLE_TE_NVFP4_FFN=1`).
Needs Blackwell sm_100+ + `transformer_engine`; else auto-falls back to BF16.
Impl: `models/dits/cosmos3video.py`, `efficiency/transforms/nvfp4_ffn.py`; kernels `jit_kernel/csrc/gemm/nvfp4`.

**Reference:** NVIDIA [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) (NVFP4 block scaling).
