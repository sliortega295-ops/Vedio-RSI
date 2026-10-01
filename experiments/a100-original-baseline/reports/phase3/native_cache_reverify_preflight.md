# Cache native objective re-verification: preflight and fail-closed design

Status at 2026-09-14 05:56 UTC: **NO CACHE DELIVERY; NOT RUN**. This is
preparation for the archived Master's independent `verify_delivery`, not
an outer acceptance or an Agent search decision. `verify_delivery.py`
source SHA is `5a8b0c09f68eeb272394be70e2926ea5a0744261fda4776de4962e2a48de7e41`
and remains unchanged. Its `reassess()` (`:666–692`) calls the executable
named by `PLAN_EVAL_PYTHON` with fixed `search/plan_eval.py --model
sana_video_2b_a100 --no-gemini --assess <real-run> --out
<real-run>/reverify_verdict.json --baseline-frames <frozen dense frames>`;
no `--no-refresh-collection` is passed for Cache. The lossy branch
(`:809–820`) requires numeric `lpips_max` after this call, while it
recomputes authoritative speed separately from the frozen 578.02 s
baseline and durable benchmark. A direct local call would use a Python
without the approved Torch/LPIPS/AlexNet stack, likely overwrite current
quality with `lpips:dependencies_missing`, and fail
`plan_eval_reverify_no_lpips`. Previous remote LPIPS backfills are **not**
this independent verifier action.

Before any native verifier runs, existing small run evidence has been
byte-preserved under
`artifacts/phase3/pre_native_verify_snapshots/20260914T055633.095853Z/`.
Receipt SHA
`6a0dc9a2300eb3d20dc775e6e963013a83b49ca600ece6cd34d821328580cd99`
is `verified_snapshot` for current real R01/R02/R03/R04 launch, manifest,
metadata, benchmark/run-config/quality/collection/assessment, and any
visual/cache evidence present. The snapshot script SHA is
`e95cb7a7bba47b59f88291ff244caea2c4ca5187ede7af0d60ef03ed593cb44f`;
it copies no video/model weight and checks source SHA before/after each copy.
R04's copied Executor visual FAIL remains marked unsupported because its
active session had no R04 image input.

The site-only transport to implement **after a genuine DELIVERY exists**:

1. An executable, coordinator-owned `PLAN_EVAL_PYTHON` shim accepts only
   the exact archived argument vector above for a real seq0002 Cache
   frontier run in the current `DELIVERY.json`; reject alternate script,
   model, paths, extra flags, output filename, or any GPU operation.
   This is scoped to the Master's `verify_delivery` invocation, not a
   general Python replacement. Preserve argv, exit, UTC/monotonic timing,
   and source SHA in a local receipt.
2. The shim sends a typed `reassess_quality` request through the
   coordinator's project-local queue. The host broker admits it from
   that queue only when the request's run/launch/manifest SHA match one
   exact current Cache `DELIVERY.json` frontier point and a reviewed
   predecessor manifest. Existing submit/status/sync/assess queue rules
   remain unchanged; no Agent-owned shell or worktree Python is executed.
3. The host checks the first quality receipt and all intervening
   pre→post SHA transitions for that run. The current benchmark, quality,
   collection, assessment, 80 frames, five videos, fixed original
   collector/judge source, offline dependencies, and frozen baseline
   must match exact accepted hashes. Any Agent rewrite, additional
   frontier point, or missing chain fails closed before remote work.
4. A trusted, SHA-locked remote CPU worker performs **one original
   collector refresh** plus original `plan_eval --no-gemini
   --no-refresh-collection` (to avoid a second refresh), with
   `CUDA_VISIBLE_DEVICES=''`, socket-deny guard, fixed Torch/LPIPS
   target and AlexNet SHA. It records selected pair indices, all five
   prompt coverage, process CPU/wall, no-network audit, and pre/post
   snapshots. Only small JSON/log/receipt files are pulled and SHA-checked.
   It writes the requested `reverify_verdict.json` in the *same real run*.
5. The shim forwards the original `plan_eval` exit status and stdout;
   it does not invent numeric LPIPS, remove the collector's separate
   Gemini blocker, or alter the archived verifier's objective rules.
   Every later reassess needs a newly reviewed post-hash/receipt chain.
   The Master must separately receive real candidate/dense image input
   and make its own lossy visual decision before integration.

This expands the earlier R01/R02-only design to whatever *actual*
DELIVERY points the original Cache Executor finally names, never to
all measured candidates by default. No shim, broker route, remote
reassess, local verifier call, or GPU action has yet occurred.
