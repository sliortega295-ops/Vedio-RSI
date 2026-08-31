# sana2b-cache_ca-0001

Run workflow `cache_ca` for model `sana_video_2b_h100`
under aspect `cache` using the experiment-local baseline copy.

The model baseline has already been materialized into this worktree from
`models/sana_video_2b_h100/model.toml`. Workflow-specific executor/reviewer prompts may
extend this goal, but runtime mutations must stay inside this experiment.


## Cache Scope

Optimize the target model by caching repeated denoising, transformer, attention,
block, residual, feature, or forecast work across denoising steps. This workflow
is for cache methods only; do not spend config work on kernel fusion,
quantization, token pruning, VAE decode, text encoders, scheduler changes,
prompt changes, or benchmark shape changes.

The editable target model source must be the copy in your experiment worktree;
locate it within the worktree and only edit inside the worktree. Do not patch
the shared reference bundle under `/lustre/.../code` or any shared checkpoint,
VAE, Hugging Face cache, or baseline run.

## Fixed Evaluation Contract

Every cache config must be judged by a full target-model run and aligned
quality assessment. Microbench, single-DiT, or module-only evidence may be used
for debugging, but it is never sufficient for workflow progress or method
judgment.

The full run must preserve the target model's official evaluation contract:

- model profile: the experiment's `model_id` (see goal.md / experiment.json);
- prompt file: the target model's validation prompt file (see the model profile
  / baseline manifest);
- prompt count: the first five prompts of the target model's validation set;
- resolution, duration, frame count, and fps: the target model's official eval
  profile (from the model profile / baseline manifest);
- denoising steps, guidance, flow shift, and motion score: the target model's
  official generation config (from the model profile / baseline manifest);
- checkpoint, VAE, scheduler, resolution, frame count, prompt text, and seed
  policy must not be changed to obtain speed or quality.

After a full run completes, preserve its benchmark, five videos or grouped
frames, and exit the executor invocation. The explicit workflow graph then runs
an independent blind Codex visual reviewer; the executor must not write that
reviewer's verdict or call Gemini itself.

The workflow gate accepts only durable `assess_verdict.json` evidence merged by
that reviewer node with
numeric `baseline_total_s`, `config_total_s`, `speedup`, numeric
`lpips_max`, `visual_provider=codex`, a complete pass-or-fail
`codex_visual_overall`, a valid `codex_visual_verdict.json`, no infrastructure
blockers, and a matching full-run config. A complete visual fail is usable
quality evidence for frontier placement; it is not automatically a workflow or
method failure.

## Closed Method Scope

The config set for this workflow is closed and contains exactly three cache
families:

1. TeaCache
2. EasyCache
3. TaylorSeer

Implement, compare, and tune only these three families. Do not introduce PAB,
DeepCache, FasterCache, Cache-DiT, generic fixed-step reuse, attention
broadcast, token pruning, or another cache family as a config. The
cache-disabled target-model run is the control, not a fourth config. A fixed
call-index reuse schedule may be used briefly to debug integration or calibrate
timing, but it cannot be retained, ranked, or presented as a workflow result.

Every measured config must belong to exactly one of the three families.
Cross-family hybrids are outside this workflow because they make attribution
and matched-time comparison ambiguous. Implementation work within a family is
allowed and expected: faithfully integrate the method into the target model, remove its
bookkeeping overhead, choose its cache payload and placement, and tune its
native refresh, threshold, history, correction, layer, and timestep controls.
Do not add an independent optimization such as kernel fusion, precision change,
quantization, step reduction, or scheduler change to make one family appear
faster.

## Matched-Time Quality Objective

The objective is not to find the fastest isolated point from each family. Find
which of TeaCache, EasyCache, and TaylorSeer preserves the most quality at the
same measured inference-time compression, and find the best implementation and
parameters for each useful shared time budget.

Use full-run end-to-end wall time from the fixed evaluation contract:

```text
time_ratio = config_total_s / baseline_total_s
time_compression = 1 - time_ratio
speedup = baseline_total_s / config_total_s
```

All runs in one comparison must use the same baseline, hardware/job shape, and
fixed inference contract. Two operating points are matched only when their
measured `time_ratio` values differ by at most 2% relative. If they are not
matched, tune and rerun the family at the target budget; do not compare quality
at unequal speeds and do not estimate full-run speed by multiplying module or
microbenchmark speedups. For noisy or close results, repeat the same point and
use the median full-run time.

Build an evidence-backed speed/quality curve for each family, then choose shared
time targets from the overlap of the three feasible ranges. At every shared
target, compare complete outputs over all five prompts using aligned LPIPS and
the independent blind Codex visual assessment, including temporal consistency
and visible artifacts. Rank quality only after the matched-time condition is
met. Report the Codex visual judgment and prompt-level failures together with
LPIPS mean/max when available;
do not hide a bad prompt behind an aggregate. Time is a constraint for this
comparison, not a quality score or a tie-breaking advantage.

The final result must identify the quality winner and exact recipe at each
matched target, plus the overall Pareto frontier. If a family cannot reach a
target faithfully, record its measured feasible range and do not claim an
unmatched winner.

The repo already has cache-method search material:

- `search_space/01_cache.md` describes several cache directions. Only its
  TeaCache-style timestep-aware reuse, EasyCache-style runtime-adaptive
  transform-vector reuse, and Taylor-style forecasting material is in scope.
  Ignore the other families in that document for this workflow.
- `config/step_cache/` contains existing cache family manifests such as
  TeaCache-style signal reuse and several out-of-scope methods. Existing code
  is reference material only and does not expand the closed config set.
- Historical Hunyuan/Cosmos artifacts contain a generic TeaCache controller and
  prior quality-gated runs, but those are references only. Adapt the idea to
  the target model's actual denoising/transformer path instead of copying a
  different model adapter blindly.

For each config, record the signal source, reuse payload, refresh rule,
hit/recompute pattern, OFF path, full-run runtime, normalized `time_ratio`,
LPIPS, Codex visual verdict, and failure mode. Also record the complete method
parameter point, its parent trial, target time budget, and why the previous
full-evaluation evidence selected the next adjustment. A cache method is not
proven by speed alone, and a quality winner is not proven at unmatched speed.

## Execution round limit

You have a hard budget of **20 optimization rounds** (config attempts) for this workflow. This is an execution-round limit, not a suggestion: pace your search so that by round 20 you have finalized and delivered your best retained frontier. Do not plan for more than 20 rounds. One round = one config implemented, launched, evaluated, and gated.


## Optimization loop, gating, and delivery contract

You are ONE executor sub-agent. You optimize a SINGLE technique for the target
model inside your own experiment worktree. A separate master orchestrator agent
started you, will independently re-verify your delivery, and will resume you
with corrections if it finds problems. There is no other agent between you and
the master.

### Baseline (frozen — never re-run it)

The frozen baseline is given in the "Frozen baseline" block appended below
(numbers + `run_dir` + `baseline_frames` + `timing_scope` + `model_id`). It was
measured once for the whole experiment. Do NOT run or re-measure the baseline.
Measure every config against it with the SAME timing scope.

### Execution round limit (hard budget)

Your technique scope states your exact hard round budget — follow that number
(it governs; it may differ per technique). One round = one config implemented,
launched, evaluated, and gated. Pace your search so that by the final round you
have finalized and delivered your best retained frontier, and **deliver early if
your frontier plateaus**. Do not plan beyond the budget your scope states.

### Each round

1. Hypothesize one improvement (avoid a previously recorded failure signature).
2. Implement exactly ONE config by editing the target model's inference code
   **inside your experiment worktree only** (locate it in the worktree; do not
   edit anything outside the worktree). Keep the technique's semantics invariant
   (do not change scheduler/step count/resolution/frames/guidance/etc.).
3. Launch: `python scripts/launch_config.py <your-config>.toml --mode sbatch --confirm-submit`
4. Collect when the job finishes: `python scripts/collect_run.py runs/<run-id>`
5. GATE — branch on the correctness mode in your technique scope; use no external
   vision API:
   - **Lossy / quality-gated (`cache`, `pisa`)**: run
     `"$PLAN_EVAL_PYTHON" search/plan_eval.py --model sana_video_2b_h100 --no-gemini --assess runs/<run-id> --baseline-frames /home/jiangzhikun/yongyan_liu/Experiments/SolAgent/20260831-sana-video-2b-full-exploration/artifacts/baseline/sana2b-baseline_bl-0001/worktree/runs/20260831-091727-sana_video_2b_h100_dense_baseline-formal-dense-retry3/outputs/frames --out runs/<run-id>/assess_verdict.json`
     (the preset eval-environment Python) for speedup + aligned LPIPS. Then use
     YOUR OWN built-in multimodal vision to inspect config frames beside the
     frozen baseline for both authenticity and new artifacts, following
     `evals/rubrics/gemini_visual_artifact_gate.md`. Write
     `runs/<run-id>/visual_verdict.json` with `overall`, `max_severity`,
     `artifacts`, and `note`.
   - **Lossless / correctness-defined (`kernel`, `topology`)**: run
     `"$PLAN_EVAL_PYTHON" search/plan_eval.py --model sana_video_2b_h100 --no-gemini --no-refresh-collection --assess runs/<run-id> --out runs/<run-id>/assess_verdict.json`
     with NO `--baseline-frames`. This reuses the durable benchmark for the speed
     report and must not compute an output-difference metric. Inspect frames only
     to confirm authenticity (a real run of this config, not a resubmitted or
     mismatched baseline); do not judge visual quality. Do NOT compare outputs at
     all—no bit/latent/fp-tolerance/LPIPS comparison.
6. Retain in your frontier only when the config is authentic, OFF-identity
   holds when disabled, and latency or peak memory measurably improves. For a
   lossy technique, aligned LPIPS and your visual-quality verdict must also pass.
   For a lossless technique, instead reason about the actual method: it computes
   the same algorithm; preserves denoising-step and global logical DiT/model-call
   counts; and introduces no approximation, step skip, sparsity, sub-16-bit
   quantization, rank reduction, or changed logical work. Record that argument
   and the counts as correctness evidence. Numeric output movement is never a
   reason to reject a lossless config.

### Delivery

By your technique scope's final round (or on genuine convergence) write
`DELIVERY.json` at your worktree root with:

```json
{
  "schema_version": 2,
  "status": "complete",
  "component": "<kernel|cache|pisa|topology>",
  "model_id": "sana_video_2b_h100",
  "baseline": { "total_s": <frozen>, "run_dir": "<frozen run_dir>", "timing_scope": "<...>" },
  "frontier_points": [
    {
      "config_id": "<id>",
      "run_dir": "runs/<run-id>",
      "activation": { "env": { "<activation env var>": "<value>" } },
      "implementation_manifest": { "path": "config/<id>.toml", "sha256": "<...>" },
      "performance": { "frontier_axis": "latency|peak_memory", "baseline_total_s": <frozen>, "config_total_s": <measured>, "speedup": <computed> },
      "quality": { "mode": "quality_gated|not_gated", "lpips_max": <number-or-null>, "lpips_mean": <number-or-null>, "visual_overall": "pass|fail|authenticity_only", "visual_verdict": "runs/<run-id>/visual_verdict.json", "relation": "equivalent|better|worse|not_applicable" },
      "artifacts": ["runs/<run-id>/outputs/out.mp4", "runs/<run-id>/outputs/frames", "runs/<run-id>/assess_verdict.json", "runs/<run-id>/visual_verdict.json", "runs/<run-id>/outputs/benchmark.json"]
    }
  ],
  "pareto_assessment": "<short note>"
}
```

Every `frontier_point` MUST reference a REAL `run_dir` that actually exists and
contains a real `out.mp4`, `benchmark.json`, and `assess_verdict.json` from a
real GPU run. **Do not fabricate or misreport numbers.** The master orchestrator
will independently recompute performance against the frozen baseline, verify
run provenance, and view config frames for authenticity. It applies LPIPS
and visual-quality gates only to lossy techniques; for lossless techniques it
instead audits structural evidence and the method's mathematical equivalence.
Any fabricated/mismatched video, misreported performance, failed applicable
gate, or malformed delivery will be rejected and you will be resumed to fix it.

Write `DELIVERY.json` as your final action.


## Frozen baseline (do not re-run)

```json
{
  "model_id": "sana_video_2b_h100",
  "total_s": 61.7,
  "denoise_s": null,
  "timing_scope": "warm_single_prompt_gen.generate_including_text_encoder_denoise_vae_decode_and_video_write_excluding_model_load_and_one_step_warmup",
  "peak_memory_mib": 17770.0,
  "run_dir": "/home/jiangzhikun/yongyan_liu/Experiments/SolAgent/20260831-sana-video-2b-full-exploration/artifacts/baseline/sana2b-baseline_bl-0001/worktree/runs/20260831-091727-sana_video_2b_h100_dense_baseline-formal-dense-retry3",
  "baseline_frames": "/home/jiangzhikun/yongyan_liu/Experiments/SolAgent/20260831-sana-video-2b-full-exploration/artifacts/baseline/sana2b-baseline_bl-0001/worktree/runs/20260831-091727-sana_video_2b_h100_dense_baseline-formal-dense-retry3/outputs/frames",
  "baseline_video": "/home/jiangzhikun/yongyan_liu/Experiments/SolAgent/20260831-sana-video-2b-full-exploration/artifacts/baseline/sana2b-baseline_bl-0001/worktree/runs/20260831-091727-sana_video_2b_h100_dense_baseline-formal-dense-retry3/outputs/out.mp4",
  "world_size": 1,
  "world_size_source": "run_artifact:run_config.json:world_size",
  "resource_envelope": {
    "nodes": 1,
    "gpus_per_node": 1,
    "world_size": 1,
    "allocated_gpus": 1,
    "hardware": null
  },
  "frozen_at": "2026-08-31T09:22:38Z",
  "source": "override_run_dir"
}
```

## Approved SANA-Video 2B executor overlay

Before taking any action, read
`models/sana_video_2b_h100/EXECUTOR_OVERLAY.md` in this worktree completely.
It is the authoritative site-specific narrowing of this recovered official
prompt wherever the two conflict. In particular: local-mode launches only,
one fixed screening prompt, no VBench or validator agent, complete
`TRAJECTORY.jsonl`, and the official 20-round Cache convergence/cap rule.
