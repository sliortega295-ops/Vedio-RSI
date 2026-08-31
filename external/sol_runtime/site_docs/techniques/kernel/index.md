# Kernel fusion (KWL)

The DiT spends a lot of time in many small ops around attention/FFN (RMSNorm,
scale/shift, residual gates, RoPE, modulation). KWL fuses them into far fewer
kernels to cut launch overhead and intermediate read/write. **Algorithm-lossless**
(only bf16 rounding differs); each fusion is shape-specialized and only kept where it
helps the target workload.

## Why fusion helps here

The expensive math in a DiT block is the two big consumers — attention and the FFN
GEMMs. But around them sits a long chain of **small, memory-bound** ops over the
`[B, S, D]` activation: RMSNorm, the AdaLN `(1+scale)·x + shift` modulation, Q/K-norm,
RoPE, residual gates, the activation. Each of these, run eagerly, is its own CUDA
kernel — so each one **launches a kernel, reads the whole activation from HBM, does a
trivial amount of arithmetic, and writes the whole activation back to HBM.** With
hundreds of thousands of video tokens, the bottleneck is this HBM round-tripping and
launch overhead, not the FLOPs.

**Kernel fusion** collapses a chain of these ops into a single kernel: load the row
once into registers/SRAM, do the whole `norm → scale/shift → gate → …` sequence there,
write once. For an N-op chain that turns `N` launches and `~N` HBM read/write passes
into **one launch and one read + one write**.

**Algorithm-lossless.** Each fused kernel reproduces the *exact* eager bf16 rounding
sequence (note in `ltx2_adaln.py` how the casts `→bf16→fp32` are replayed step by
step), so the output is bit-faithful to eager modulo the unavoidable fused-op rounding.
Every fusion is shape-specialized and gated by an env flag, so it is kept only where it
actually wins on the target workload.

## What we fuse

| Group | What is fused |
|---|---|
| [Operator fusions](fusions.md) | RMS+AdaLN, Q/K-norm+RoPE, dual / CA-dual modulation, all-9 Ada values, residual gate, FFN proj_in+GELU, audio QKVG, VAE GroupNorm+SiLU |
| [Compile & VAE](compile.md) | gate-to-out subgraph compile, tiled-VAE-decoder compile, torch.compile modes |
| [Branch sharing](sharing.md) | CFG/STG block-0 self-attn sharing, guidance-prefix sharing |

Impl: `jit_kernel/diffusion/{triton,cutedsl}/`; wiring `efficiency/transforms/kwl_fusions.py`
(env `SGLANG_HQ_KWL_*` → `SGLANG_LTX2_{FUSED_*,SHARE_*,COMPILE_*}`). Used by **LTX-2.3**.
