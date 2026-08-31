# MixDQ

**MixDQ** targets **few-step** text-to-image diffusion (SDXL-Turbo-style), where there
are too few steps to hide quantization error and the text-embedding path is extremely
sensitive. It contributes a BOS-aware handling of the highly sensitive text embeddings,
a metric-decoupled sensitivity analysis per layer, and an integer-programming bit-width
allocator. The result is mixed-precision down to W3.66A16 / W4A8 with negligible quality
or text-alignment loss, giving 3–4× memory reduction and ~1.5× latency.

**Relevance here.** MixDQ is the canonical example of *mixed-precision* allocation —
keep the few most sensitive layers/steps high-precision, push the rest low. That is the
same principle behind our step-selective NVFP4 (sensitive steps stay BF16).

**Paper:** [MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed Precision Quantization (arXiv:2405.17873)](https://arxiv.org/abs/2405.17873) · [code](https://github.com/A-suozhang/MixDQ)
