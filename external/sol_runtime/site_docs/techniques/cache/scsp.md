# SCSP stage-1 step-skip

A fixed, schedule-based step-skip used for **LTX-2.3 stage 1**: a preset list of
stage-1 denoise calls is skipped (their residual replayed) rather than deciding
per-step at runtime. The shipped preset is `8of15_last_29calls`.

**In this repo.** LTX `fullopt`'s cache component. Impl:
`runtime/cache/ltx2_stage1_cache_core.py`. Env:
`SGLANG_LTX2_STAGE1_CACHE_CORE_{ENABLED,PRESET}`.

(Internal schedule — no external paper.)
