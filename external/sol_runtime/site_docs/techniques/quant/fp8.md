# FP8

**FP8** is the 8-bit floating-point interchange format (E4M3 / E5M2) standardized by
NVIDIA/Arm/Intel — a near-drop-in 2× memory/throughput win over BF16 for linears,
with E4M3 used for weights/activations and E5M2 where wider range is needed.

**In this repo.** Provided as a quantization backend (`runtime/layers/quantization/fp8.py`,
`modelopt_fp8.py`); not on a default video line (the video lines use NVFP4).

**Paper:** [FP8 Formats for Deep Learning (arXiv:2209.05433)](https://arxiv.org/abs/2209.05433)
