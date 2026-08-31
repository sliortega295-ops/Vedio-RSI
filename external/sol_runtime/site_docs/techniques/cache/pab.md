# PAB (Pyramid Attention Broadcast)

**PAB** exploits the observation that the attention difference across diffusion
steps follows a U-shaped pattern (lots of redundancy in the middle). It **broadcasts**
an attention output to subsequent steps in a pyramid fashion, with different
broadcast ranges per attention type (spatial / temporal / cross) according to their
variance — skipping redundant attention recomputation.

**In this repo.** Provided (`runtime/cache/ltx2_pab.py`); not on a default line.

**Paper:** [Real-Time Video Generation with Pyramid Attention Broadcast (arXiv:2408.12588)](https://arxiv.org/abs/2408.12588) · [project](https://oahzxl.github.io/PAB/)
