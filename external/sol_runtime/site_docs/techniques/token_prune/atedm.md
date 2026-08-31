# AT-EDM — Attention-Driven pruning

**AT-EDM** is a *pruning* (not merging) method: it drops redundant tokens at run time
using the model's own attention maps as the importance signal. It introduces
**Generalized Weighted PageRank (G-WPR)** — treat the attention matrix as a graph and
rank tokens by their PageRank-style centrality, so tokens that many others attend to are
kept. Pruned tokens are later restored with a **similarity-based recovery** so that
convolution / spatial ops still see a full feature map. Fully **training-free**.

- **Importance signal:** attention-graph centrality (G-WPR over attention maps).
- **Recovery:** similarity-based token restoration before spatial ops.
- **Reported:** ~38.8% FLOPs saving, up to 1.53× on SD-XL at near-equal FID/CLIP.

The canonical attention-importance pruning method — closest in spirit to importance-
scored pruning in video DiTs.

**Paper:** [Attention-Driven Training-Free Efficiency Enhancement of Diffusion Models (arXiv:2405.05252)](https://arxiv.org/abs/2405.05252) · [project](https://atedm.github.io/)
