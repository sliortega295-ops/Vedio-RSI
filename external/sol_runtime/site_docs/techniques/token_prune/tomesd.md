# ToMeSD — Token Merging for Stable Diffusion

**ToMeSD** brings Token Merging (originally a ViT-classification trick) to diffusion.
The observation: generated images have large local redundancy, so many tokens are
near-duplicates. Before each transformer block it partitions tokens into a source and
destination set, uses **bipartite soft matching** to merge each source token into its
most-similar destination token, runs attention/FFN on the reduced set, then **un-merges**
(copies the merged result back to all the original positions). It is **training-free**
and the merge ratio is a single knob.

- **Importance signal:** token-to-token cosine similarity (merge the most similar).
- **Recovery:** un-merge — the representative token's output is broadcast back.
- **Reported:** up to 60% token reduction, ~2× faster, up to 5.6× less memory on SD,
  with little quality loss.

The foundational method every later diffusion token-reduction work builds on.

**Paper:** [Token Merging for Fast Stable Diffusion (arXiv:2303.17604)](https://arxiv.org/abs/2303.17604) · [code](https://github.com/dbolya/tomesd)
