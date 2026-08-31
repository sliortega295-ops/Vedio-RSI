# MXFP4 / MXFP8

**Microscaling (MX)** formats pair a per-block shared scale with narrow per-element
floating-point (MXFP4 = 4-bit, MXFP8 = 8-bit). The block scale recovers dynamic range
lost to the narrow element type, making sub-8-bit inference/training practical with
low accuracy loss. Standardized by the Open Compute Project.

**In this repo.** Provided (`runtime/layers/quantization/mxfp4.py`, `mxfp4_npu.py`,
`mxfp8_npu.py`, `modelslim_mxfp4_scheme.py`, `modelslim_mxfp8_scheme.py`); not on a
default video line.

**Paper:** [Microscaling Data Formats for Deep Learning (arXiv:2310.10537)](https://arxiv.org/abs/2310.10537)
