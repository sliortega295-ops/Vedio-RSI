# Video token reduction

Video adds a temporal axis: tokens are redundant not only within a frame but across
frames, and reduction must avoid flicker (temporal coherence) on top of preserving
spatial detail. Most image methods (ToMe, AT-EDM) are attention-importance based; the
video work below adapts the importance signal and recovery to the spatiotemporal
setting.

| Method | Idea | Paper |
|---|---|---|
| **Importance-Based Token Merging** | merge by a learned/derived importance rather than raw similarity, for both image and video | [2411.16720](https://arxiv.org/abs/2411.16720) |
| **Temporal-Aware Pruning** | prune with an explicit temporal-coherence criterion so frames stay flicker-free | [2605.17837](https://arxiv.org/abs/2605.17837) |
| **FastSTAR** | spatiotemporal token pruning for autoregressive video synthesis | [2603.07192](https://arxiv.org/abs/2603.07192) |
| **Token Pruning for In-Context Generation** | prune the long reference/context tokens that in-context video DiTs append | [2602.01609](https://arxiv.org/abs/2602.01609) |

**Common theme.** Naïve per-frame pruning breaks temporal consistency; these methods
either score importance with temporal context or recover dropped tokens from adjacent
frames/steps so motion stays smooth — the same coherence concern that makes the
*middle* denoise steps the safest place to prune in a video refine stage.
