# EasyCache

**EasyCache** is a calibration-free, runtime-adaptive cache: it estimates the
relative change of the transformation between steps (on a spatially subsampled
tensor for cheapness) and reuses the previous computation when the change is below a
threshold. No offline profiling or precomputation.

**In this repo.** SANA-Video's cache (part of its `EasyCache + fusion + compile`
line). Driven from the SANA DiT (`models/dits/sana_video.py`). Flags:
`--easycache <thr> --ec-warmup --ec-subsample` (env `SGLANG_SANA_EASYCACHE_{THRESH,WARMUP,SUBSAMPLE}`).

**Paper:** [Less is Enough: Training-Free Video Diffusion Acceleration via Runtime-Adaptive Caching (arXiv:2507.02860)](https://arxiv.org/abs/2507.02860) · [code](https://github.com/H-EmbodVis/EasyCache)
