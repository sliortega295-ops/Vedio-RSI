# ToDo — Token Downsampling

**ToDo** extends ToMe but removes its two weaknesses: the bipartite-matching step is
itself costly, and aggressive merging hurts quality at high resolution. Instead of
similarity matching, ToDo **downsamples** the key/value tokens with a strided/grid
pooling — queries stay full, but each query attends over a downsampled K/V set. That
makes attention cheaper while keeping every output position, so there is no un-merge and
no matching overhead.

- **Importance signal:** none learned — a fixed spatial downsampling of K/V.
- **Recovery:** not needed — Q (and thus every output token) is preserved.
- **Reported:** up to ~2× at common sizes, ~4.5×+ at 2048×2048, training-free.

The simplest high-resolution-friendly point in the merge family.

**Paper:** [ToDo: Token Downsampling for Efficient Generation of High-Resolution Images (arXiv:2402.13573)](https://arxiv.org/abs/2402.13573) · [code](https://github.com/ethansmith2000/ImprovedTokenMerge)
