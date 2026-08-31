# DaTo — token pruning ⨯ caching

**DaTo** combines the two orthogonal redundancies diffusion exposes: **across tokens**
(spatial, like ToMe/AT-EDM) and **across denoise steps** (temporal, like
[cache](../cache/index.md)). It prunes tokens within a step *and* reuses cached features
across steps, so the two savings multiply instead of competing. Training-free, it
reports up to **9× acceleration on Stable Diffusion** on ImageNet.

- **Importance signal:** similarity-based token selection + cross-step feature reuse.
- **Recovery:** cached features fill in both pruned tokens and skipped steps.
- **Why it matters here:** the clearest demonstration that token reduction and step
  caching stack — relevant to pipelines that already run a cache (TeaCache / SCSP).

**Paper:** [Token Pruning for Caching Better: 9× Acceleration on Stable Diffusion for Free (arXiv:2501.00375)](https://arxiv.org/abs/2501.00375)
