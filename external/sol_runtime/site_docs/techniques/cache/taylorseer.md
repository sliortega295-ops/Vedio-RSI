# TaylorSeer

Most caches **reuse** a stale feature; **TaylorSeer** instead **forecasts** it.
It approximates higher-order time-derivatives of a feature from its values at
earlier steps and predicts the feature at the current step via a Taylor-series
expansion — reducing the error that plain reuse accumulates over large step gaps.

**In this repo.** Available through the cache-dit integration
(`runtime/cache/cache_dit_integration.py`), as one of the cache-dit policies; not on
a default model line.

**Paper:** [From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers (arXiv:2503.06923)](https://arxiv.org/abs/2503.06923) · [code](https://github.com/Shenyi-Z/TaylorSeer)
