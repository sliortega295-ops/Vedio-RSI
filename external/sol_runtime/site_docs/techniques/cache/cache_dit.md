# Cache-DiT (DBCache / TaylorSeer / FBCache)

Integration with the **cache-dit** library, which exposes several block-level cache
policies — most notably **DBCache** (Dual-Block Cache, caches across a window of
transformer blocks), **first-block cache (FBCache)**, and **TaylorSeer** forecasting.
Useful for the Wan / FLUX / Hunyuan / Qwen-Image family that already register a
cache-dit block adapter.

**In this repo.** `runtime/cache/cache_dit_integration.py` + `ltx2_block_adapter.py`.
Enabled via `SGLANG_CACHE_DIT_*`. Not on the three default lines.

**Paper:** TaylorSeer — [arXiv:2503.06923](https://arxiv.org/abs/2503.06923). DBCache / FBCache are library features of cache-dit.
