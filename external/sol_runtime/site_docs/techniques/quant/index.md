# Quantization

Run the heavy linear layers below BF16 to make the GEMMs faster / smaller. The
methods differ in bit-width and how they preserve accuracy.

## Formats & engines in this repo

| Method | Bits | Idea | Used by |
|---|---|---|---|
| [NVFP4](nvfp4.md) | 4 (FP4, block-scaled) | TE NVFP4, step-selective | Cosmos3, LTX |
| [FP8](fp8.md) | 8 (E4M3 / E5M2) | drop-in 8-bit | (provided) |
| [MXFP4 / MXFP8](mxfp.md) | 4 / 8 (microscaling) | per-block shared scale | (provided) |
| [SVDQuant (Nunchaku)](svdquant.md) | 4 | low-rank absorbs outliers | (provided) |

All are **lossy**; the NVFP4 line keeps the most quality-sensitive denoise steps in BF16.

## Algorithms & background

The diffusion/DiT PTQ literature these formats build on. Two themes recur and directly
motivate our step-selective NVFP4: (1) error varies strongly across *timesteps* and
*tokens*, and (2) a few sensitive layers/steps must stay high-precision (mixed precision).

| Method | Scope | Idea |
|---|---|---|
| [SmoothQuant](smoothquant.md) | building block | migrate activation outliers into weights (per-channel scale) |
| [ViDiT-Q](viditq.md) | video DiT | token- & timestep-wise dynamic quant + mixed precision |
| [Q-DiT](qdit.md) | image DiT | fine-grained group-wise quant with searched groups |
| [MixDQ](mixdq.md) | few-step T2I | BOS-aware + integer-program mixed-precision allocation |
