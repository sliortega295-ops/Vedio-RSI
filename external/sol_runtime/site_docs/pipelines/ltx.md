# LTX-2.3 1080p/10s

LTX-2.3 is the HQ two-stage pipeline (single GPU): stage 1 denoises a low-res latent
(30 steps), stage 2 does a 3-sigma refinement with a distilled LoRA merged in. It is
the **reference for composing all five acceleration methods at once**, each owning a
distinct seam so they don't interfere.

## Acceleration line (`fullopt`): the full five-method stack

| Component | Where | Type | Paper / details |
|---|---|---|---|
| [KWL kernel fusion](../techniques/kernel/index.md) | both stages (DiT + VAE) | lossless | ~13 fused ops + compile |
| [Stage-1 SCSP cache](../techniques/cache/scsp.md) | stage 1 | cache (lossy) | preset `8of15_last_29calls` |
| [PISA sparse attention](../techniques/sparse/pisa.md) | stage 2 (`transformer_2`) | sparse (lossy) | [arXiv:2602.01077](https://arxiv.org/abs/2602.01077) |
| [NVFP4 video FFN](../techniques/quant/nvfp4.md) | both stages | quant (lossy) | TransformerEngine |
| [Stage-2 token-prune](../techniques/token_prune/index.md) | stage 2 midpoint | prune (lossy) | feat-norm, keep 0.5 |

**Why these compose.** Each method owns a different seam: KWL is lossless kernel work;
the cache acts on stage-1 *steps*; PISA owns the stage-2 *attention backend*; NVFP4
owns the *FFN precision*; token-prune owns the stage-2 *token set* at the midpoint.
Because their scopes are disjoint, the `runtime/efficiency/` framework can compose
them and each stays off==identity when disabled.

## Design notes

- Official two-stage HQ: 1088×1920, 241f, 24fps, stage-1 30 steps + stage-2 3-sigma
  refine (`0.909375, 0.725, 0.421875`), guidance 3.0, official distilled LoRA merged
  for stage 2.
- `WARMUP=true` by default so the one-time torch.compile/autotune cost lands in a
  warmup pass, not the timed run.
- `fullopt` is **self-contained** — it bakes the whole recipe (KWL env, stage-1 cache
  preset, stage-2 PISA backend config, NVFP4 FFN, token-prune env). No extra env.

## Run

```bash
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh baseline   # dense two-stage reference
bash scripts/ltx/run_ltx23_sglang_hq_1080p10s.sh fullopt    # the 5-method stack
```

## Measured (1088×1920, 241f, GB200, warmup-excluded)

| config | warm | speedup |
|---|---|---|
| baseline (dense two-stage) | 95.7 s | 1.00× |
| fullopt (5-method stack) | 39.2 s | **~2.4×** |
