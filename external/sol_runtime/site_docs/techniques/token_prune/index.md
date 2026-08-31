# Token pruning

Diffusion over images/video carries large spatial redundancy: neighbouring tokens
(patches) often encode near-identical content, and not every token needs full
processing at every denoise step. **Token reduction** exploits this — it shrinks the
sequence the heavy block stack runs on, then restores full resolution afterward. Like
[sparse attention](../sparse/index.md) and [cache](../cache/index.md), it is an
algorithm-level approximation (lossy), and like them it composes as a backend/runtime
swap rather than a weight change. It differs from kernel fusion, which is lossless and
operates *below* the algorithm.

This page surveys the training-free literature. Approaches divide into three families
by *what* they do with the redundant tokens and *how* they pick them.

## Taxonomy

| Family | Mechanism | Restore step |
|---|---|---|
| **Merge** | combine similar tokens into one, run fewer, then un-merge / copy back | un-merge (broadcast representative) |
| **Prune** | drop low-importance tokens entirely, run the rest | scatter-back / similarity recovery |
| **Hybrid** | combine token reduction with [caching](../cache/index.md) across steps | both |

The key design axes are: the **importance signal** (attention scores, feature norm,
similarity), *where* the reduction happens (which layers / which denoise steps), and how
dropped tokens are **recovered** so spatial/temporal consistency survives.

## Methods (training-free, open-source)

| Method | Family | Idea | Paper · code |
|---|---|---|---|
| [ToMeSD](tomesd.md) | merge | bipartite matching merges redundant tokens in SD | [2303.17604](https://arxiv.org/abs/2303.17604) · [code](https://github.com/dbolya/tomesd) |
| [ToDo](todo.md) | merge / downsample | downsample K/V tokens; fixes ToMe's cost & quality at high-res | [2402.13573](https://arxiv.org/abs/2402.13573) · [code](https://github.com/ethansmith2000/ImprovedTokenMerge) |
| [AT-EDM](atedm.md) | prune | attention-graph importance (G-WPR) + similarity recovery | [2405.05252](https://arxiv.org/abs/2405.05252) · [project](https://atedm.github.io/) |
| [DaTo](dato.md) | hybrid | token pruning + feature caching together (9× on SD) | [2501.00375](https://arxiv.org/abs/2501.00375) |
| [Video methods](video.md) | merge / prune | importance-based merge & spatiotemporal prune for video | [2411.16720](https://arxiv.org/abs/2411.16720) · others |

See each page for the importance signal, recovery method, and reported speedup.
