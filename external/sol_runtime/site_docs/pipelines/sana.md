# SANA-Video

SANA-Video is the lightest of the three (a 2B linear-attention DiT, single GPU). Its
acceleration line is deliberately **non-quantized and dense** — the model is small
enough that the wins come from *skipping steps* and *fusing kernels*, not from
low-precision GEMMs or sparse attention.

## Acceleration line: EasyCache + fusion + compile

| Component | What it does | Type | Contribution |
|---|---|---|---|
| [EasyCache](../techniques/cache/easycache.md) | runtime-adaptive step-skip (reuse transformation vectors) | cache (lossy) | the bulk of the speedup |
| `linattn-bf16` | bf16 linear-attention KV aggregation | fusion (lossless) | per-step compute |
| `qkv-merge` | one merged QKV GEMM for self-attention | fusion (lossless) | per-step compute |
| [torch.compile](../techniques/kernel/index.md) | inductor graph compile of the DiT | kernel (lossless) | ~1.3× on top |

**Why this set.** EasyCache is calibration-free, so it needs no per-prompt profiling
— ideal for an interactive 2B model. The two fusions are bit-lossless and remove
launch/intermediate overhead in the linear-attention path. `torch.compile` then folds
the remaining elementwise work into Inductor kernels.

## Compile mode (important)

The generic compile default is `max-autotune`, whose **in-process GEMM/conv autotune
deadlocks at `cuda.synchronize()`** on a *cold* Inductor cache (a grouped-conv Triton
template hangs). The entry therefore defaults `--compile` to the safe inductor
`default` mode (runs cold anywhere). For peak speed pass `--max-autotune`, which runs
autotune in a **subprocess** (per-choice timeout skips the hanging conv) and persists
the Inductor cache — the first cold run warms it, every run after is fast.

## VAE note

480p uses Wan2.1-VAE, 720p uses LTX-2-VAE — different latent denorm; the pipeline
config applies the correct one per resolution.

## Run

```bash
PY=.conda/ltx23/bin/python
# baseline (dense)
$PY scripts/sana/sana_video_sglang_run.py --output sana_baseline
# fullopt = EasyCache + fusion + compile (safe 'default' mode, ~2.1x, cold-safe)
$PY scripts/sana/sana_video_sglang_run.py --output sana_fullopt \
    --easycache 0.1 --linattn-bf16 --qkv-merge --compile
# peak (~2.56x once warm): add --max-autotune
```

## Measured (480p, 832×480, 81f, 50 steps, GB200, warmup-excluded)

| config | warm | speedup |
|---|---|---|
| baseline (dense) | 28.5 s | 1.00× |
| EasyCache + fusion + compile (`default`) | 13.5 s | **2.10×** |
| + `--max-autotune` (warm cache) | 11.0 s | **2.56×** |

The cache + fusion alone give ~1.59×; compile adds the rest.
