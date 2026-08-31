# Cache (step-skip)

Adjacent denoising steps often change the latent very little, so a cache reuses a
previous step's transformer output/residual instead of recomputing — skipping the
heavy DiT forward on "easy" steps. The variants differ in **how they decide to skip**.

| Method | Decision policy | Calibration | Used by |
|---|---|---|---|
| [TeaCache](teacache.md) | accumulated rel-L1 of timestep-modulated input vs threshold | optional (polynomial) | Cosmos3 |
| [EasyCache](easycache.md) | runtime-adaptive reuse of transformation vectors | none | SANA-Video |
| [TaylorSeer](taylorseer.md) | Taylor-expansion **forecast** of future features | none | (provided) |
| [SCSP step-skip](scsp.md) | fixed schedule on stage-1 steps | none | LTX-2.3 |
| [PAB](pab.md) | broadcast attention outputs (U-shaped redundancy) | none | (provided) |
| [Cache-DiT](cache_dit.md) | DBCache / TaylorSeer via the cache-dit library | varies | (provided) |

All cache methods are **lossy** — too-aggressive skipping drops detail; the threshold
is the quality dial.
