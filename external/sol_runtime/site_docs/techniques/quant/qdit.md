# Q-DiT

**Q-DiT** is a PTQ method for **image** diffusion transformers (DiT/PixArt-style). It
observes that DiT weights and activations have highly uneven variance across channels
and across timesteps, which coarse per-tensor quantization cannot capture. Q-DiT uses
fine-grained **group-wise** quantization with automatically searched group sizes, plus
a timestep-aware sampling of the calibration set, to push DiTs to W4A8 / W6A6 with
small FID change.

**Relevance here.** Group-wise scaling is exactly the granularity NVFP4/MXFP achieve in
hardware (block-scaled FP4) — Q-DiT is the algorithmic argument for why block scales,
not per-tensor scales, are what make 4-bit DiTs viable.

**Paper:** [Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers (arXiv:2406.17343)](https://arxiv.org/abs/2406.17343) · [code](https://github.com/Juanerx/Q-DiT)
