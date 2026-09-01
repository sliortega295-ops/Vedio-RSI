## H100 FP8 Executor Objective

Implement and search selective W8A8 E4M3 execution for the target model. This
executor owns only sub-16-bit precision policy: module selection, FP8 packing,
activation scaling, dense block guards, backend/fallback behavior, and the
quality/performance evidence needed to retain one FP8 point. Kernel fusion,
step cache, sparse attention, topology, prompt, scheduler, and workload settings
belong to other components and must remain at their frozen OFF settings during
isolated FP8 attribution runs.

For SANA-Video, begin with `GLUMBTempConv.conv_inverted` and `conv_point` because
they are 1x1 projections and are exactly expressible as last-dimension GEMMs.
Keep the depthwise and temporal convolutions, attention, embeddings, text
encoder, VAE, and output head in their original precision. Do not call the run
FP8 unless durable `SANA_FP8_INSTALL` and `SANA_FP8_MODULE_ACTIVE` receipts prove
that native E4M3 weights and GEMMs executed; BF16 fallback is a blocker or
structured negative, never a speed result.

FP8 is quality-gated, not mathematically lossless. OFF must retain the original
BF16 modules. Before a full generation, run the supplied component smoke and
require finite outputs, cosine similarity >= 0.995, and relative RMSE <= 0.10.
For full generation, preserve the fixed model, seed, prompts, dimensions, CFG
structure, denoising steps, and video contract. Inspect first/middle/last frames
against the frozen baseline and use the existing no-external-API LPIPS path when
available. The intended reproduction is bounded: Prompt 1 plus one confirmation
or Prompt 2 is sufficient; do not expand into a broad VBench campaign.

Measure FP8 first as an isolated technique against the frozen dense baseline.
The master, not this executor, composes the retained FP8 activation with Kernel
and Cache. A module microbenchmark is only a screening tool; retention requires
a matching full-generation result. Prefer a >=2% latency improvement to avoid
promoting noise. If quality needs repair, try dense first/last-block guards
before broadening quantization. Do not quantize attention or output heads unless
profile evidence ranks them above guarded FFN refinement.

### Candidate loop and hard budget

Use at most **6 rounds**, and stop early on a real plateau:

1. all-block FFN 1x1 FP8 with strict no-fallback evidence;
2. an identical confirmation if the first point is positive, otherwise diagnose
   backend/layout overhead;
3. a first/last one-block BF16 guard if quality needs repair;
4. a first/last two-block guard only if round 3 has a clear quality/speed tradeoff;
5. one profile-justified refinement, not a blind all-linear expansion;
6. final confirmation of the best retained point.

Record every attempt in `TRAJECTORY.jsonl` and `FP8-SEARCH-STATE.json` with
config/commit identity, converted modules, backend, fallback count, component
smoke, full latency, video validity, quality judgment, and retain/reject reason.

### Delivery

Deliver exactly one best measured point in `DELIVERY.json`, or an empty frontier
with `status="structured_negative"` and the real evidence showing why no FP8
point was promotable. A delivered point must include the config, run directory,
benchmark, FP8 install/activation receipt, smoke report, video/frames, quality
assessment, and matching baseline timing scope. Set `component` to `fp8` and
`quality.mode` to `quality_gated`.
