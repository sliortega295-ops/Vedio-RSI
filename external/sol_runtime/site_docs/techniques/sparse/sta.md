# STA (Sliding Tile Attention)

**STA** exploits that attention in pretrained video DiTs concentrates in localized 3D
windows. Instead of token-wise sliding windows, it attends **tile-by-tile** with a
hardware-aware sliding-window design, giving an efficient 2D/3D local attention with
high MFU and large speedups over FlashAttention — training-free, with optional
fine-tuning for more.

**In this repo.** `runtime/layers/attention/backends/sliding_tile_attn.py`.

**Paper:** [Fast Video Generation with Sliding Tile Attention (arXiv:2502.04507)](https://arxiv.org/abs/2502.04507)
