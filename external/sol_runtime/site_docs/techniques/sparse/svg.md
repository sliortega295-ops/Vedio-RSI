# Sparse VideoGen (SVG / SVG2)

**SVG** observes that attention heads split into *spatial heads* (only intra-frame
tokens matter) and *temporal heads* (only cross-frame tokens matter), profiles each
head's type online, and applies the matching sparse pattern. **SVG2** replaces
position-based clustering with **semantic-aware permutation** (k-means reorders tokens
by similarity) plus top-p dynamic budget, improving the quality/efficiency frontier.

**In this repo.** `runtime/layers/attention/backends/sparse_video_gen_2_attn.py`.

**Papers:** SVG — [arXiv:2502.01776](https://arxiv.org/abs/2502.01776); SVG2 — [arXiv:2505.18875](https://arxiv.org/abs/2505.18875) · [project](https://svg-project.github.io/)
