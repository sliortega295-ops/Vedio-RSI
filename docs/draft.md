# Cache search task contract

## Objective

Reproduce the Cache branch of the archived Sol-Video-Agent search on the frozen
SANA-Video 2B dense baseline, using the one leased H100 and preserving the full
20-round search trajectory.  Search is restricted to TeaCache, EasyCache, and
TaylorSeer; all candidate runs keep the frozen prompt, seed, shape, scheduler,
precision, and timing scope.

## Frozen inputs and outputs

- Baseline: `BASELINE-LOCK.json`, 61.7 s, peak 17,770 MiB.  It is immutable and
  must never be rerun.
- Workload: one fixed model-card prompt, 832x480, 81 frames, 16 fps, 50 steps,
  guidance 6, flow shift 8, seed 42.
- Hardware: the UUID in the shared active lease; the wrapper owns the persistent
  flock and rejects foreign compute applications.
- Per candidate: a new `runs/<run-id>` with an MP4, benchmark, ffprobe/validity,
  first/middle/last frames, exact command and source hashes.
- Search ledger: contiguous `TRAJECTORY.jsonl`, including failures and rejects.

## Correctness and quality contract

- OFF identity: an unset/disabled cache family takes the existing dense path;
  no scheduler, step-count, prompt, seed, shape, precision, or guidance change.
- A cache hit may skip only the SANA transformer-block stack; patch embedding,
  timestep/text conditioning, output norm/projection, unpatchify, scheduler, VAE,
  and all 50 denoising steps remain active.
- Every real round must validate 832x480, 81 frames, 16 fps, non-empty MP4 and
  zero residual compute applications after process exit.
- First/middle/last frames are compared with the frozen baseline for identity,
  subject integrity, gross temporal drift, ghosting, deformation, flicker, and
  lighting/composition changes.  This reproduction intentionally does not use
  VBench, Gemini, an independent validator, or a five-prompt grid per round.
- Retain only measured speed/quality points; preserve regressions, no-hit points,
  invalid outputs, build failures, OOMs, and visually unacceptable points.

## Exact commands

Static build/validation before GPU use:

```bash
PY=/home/jiangzhikun/yongyan_liu/Experiments/SolAgent/20260827-official-repro/envs/sana-cu128-phase0/bin/python
PYTHONPATH=external/sol_runtime/python:/home/jiangzhikun/yongyan_liu/Experiments/SolAgent/20260827-official-repro/baseline/targets/sana-runtime-deps-v1:/home/jiangzhikun/yongyan_liu/Experiments/SolAgent/20260827-official-repro/baseline/staging/sglang-build001-clean-v2 \
  "$PY" -m unittest tests.test_sana_video_cache
"$PY" models/sana_video_2b_h100/trajectory.py validate --ledger TRAJECTORY.jsonl
```

Every measured round:

```bash
PY=/home/jiangzhikun/yongyan_liu/Experiments/SolAgent/20260827-official-repro/envs/sana-cu128-phase0/bin/python
"$PY" scripts/launch_config.py config/sana_video_2b_h100/cache/<config>.toml \
  --mode local --run-root runs --name-suffix cache-rNN
```

## Promotion criteria

A point is eligible for the frontier only when the real wrapper run is
`VALIDATED`, its exact workload/timing provenance matches the frozen baseline,
it yields at least one genuine cache hit, its total time is better than 61.7 s,
and the built-in frame review finds no material new artifact for the intended
quality tier.  The final delivery reports only real useful points and may report
an honest gap instead of inventing tiers.

## Main risks and unknowns

1. The existing SANA EasyCache path has no durable family-neutral hit summary,
   and its shared CFG residual bookkeeping is implicit.  Add explicit decisions,
   family parameters, hit caps, and source-hashed receipts without touching the
   OFF path.
2. TeaCache needs a SANA-specific timestep-modulated signal.  Use the first
   block's AdaLN-modulated, token-subsampled input and replay the cached block
   residual, with first/last guards and bounded consecutive hits.
3. TaylorSeer must forecast instead of stale-reuse.  Use first/second-order
   discrete Taylor extrapolation from real computed residual history, accounting
   for unequal compute-step spacing and damping the forecast.  Refresh decisions
   remain adaptive; no fixed call-index schedule can be retained as a result.
4. Cache-controller synchronization and sampled signal work can erase small
   savings.  Record no-hit and overhead regressions and tune subsampling rather
   than hiding them.
5. One fixed prompt cannot establish broad perceptual quality.  Conclusions are
   explicitly screening evidence; the master performs only the approved small
   near-final sanity check.

## Ranked candidate directions

1. EasyCache adaptive transformation-rate residual replay, beginning at a
   conservative threshold and then sweeping threshold, warmup, subsampling, and
   consecutive-hit cap.
2. TeaCache first-block timestep-modulated signal with identity polynomial,
   sweeping threshold/start/hit cap to cover overlapping time ratios.
3. TaylorSeer first-order then second-order residual forecasting, tuning adaptive
   threshold, history order, damping, and hit cap.
4. Revisit only measured gaps: close matched-time holes, confirm a noisy useful
   point once, and test a distinct parameter hypothesis rather than repeating an
   already rejected failure signature.

## Evidence needed per decision

Config hash and parent SHA; implementation commit/diff; static test result; exact
launcher command; run directory; cache decision trace and hit pattern; latency,
speedup, time ratio and peak memory; MP4/ffprobe/frame hashes; built-in visual
note; retain/reject/retry/stop reason.  The 20th record must stop unless the
faithful family search genuinely converges earlier.
