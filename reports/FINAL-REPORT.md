# Sol-Video-Agent SANA-Video 2B reproduction: final report

Status: **complete for the approved bounded scope**. The experiment recovered the
official archived Sol-Agent harness, froze and ran a genuine dense SANA-Video 2B
baseline on one H100, preserved complete Kernel and Cache search trajectories,
integrated the verified winners, and ran the final candidate on exactly two
distinct prompts. It does not claim the full paper table or broad quality parity.

## Authority and reproducibility

- Experiment root: `/home/jiangzhikun/yongyan_liu/Experiments/SolAgent/20260831-sana-video-2b-full-exploration/`
- Archived official harness: `d2c6407cc9b9133f3fff49fe4b561f14980d3f8b`
- Harness branch/head: `repro/sana-video-2b-full-exploration` at `1a35329a821a3e12631e23573622b18c090176bd`
- Current-Codex transport compatibility: `4aeb3e7c2275fbaa09bc337a369eb5087e95ce89`; focused tests 7/7 pass.
- cu128 compatibility is separate from the archive: commit `5bc0c43fb7fe548af4119a8831c4e286c982c71f`, exported patch SHA-256 `91eff9ae8f19a4e934fc727533b473e17ecbde706c874e4bc5a2a1da0369ecda`; minimal-import tests 4/4 pass.
- Runtime source authority: `b0b7eb4d0a7f1f46118a356485f4523cf52e96dd`.
- Model: `Efficient-Large-Model/SANA-Video_2B_480p_diffusers` revision `db5f398b13ca086d09a50ce156c20527773841b1`.
- GPU: H100 UUID `GPU-847305ce-670b-91ee-e0a9-aa3b7833df23` (physical index 7). Every GPU launch was serialized under `state/gpu/H100.lock` and bracketed by ownership checks.

The archive commit remains an inspectable Git object. Transport and cu128 changes
are separate commits/patches rather than edits hidden inside that historical
snapshot.

## Frozen dense baseline — VALIDATED

The formal baseline is not a prior optimized run. It was launched after freezing
`state/BASELINE.json` (SHA-256
`3fdafdf00554ae4bafc91bc7729c3ba3e96af4edebac809782e6d6122ed23954`)
with all optimization flags off.

- Workload: 832x480, 81 frames, 16 fps, 50 steps, guidance 6, seed 42, model-card long prompt with motion score 30, BF16 transformer/text encoder and FP32 VAE.
- Run: `artifacts/baseline/sana2b-baseline_bl-0001/worktree/runs/20260831-091727-sana_video_2b_h100_dense_baseline-formal-dense-retry3`
- Result: 61.7 s; runtime peak 17,770 MiB; nvidia-smi peak 23,319 MiB.
- Output: valid 832x480, 81 frames at 16 fps; video SHA-256 `77b788db7c9488eeaae10497659f884c4f51df9ba8def45861f312160f550ec1`.
- Exact command is preserved in that run's `outputs/command.txt`; benchmark SHA-256 is `b2ce44dbbef06517d205f1d34e8b3cfb9c7e28bc6c3bd9d9c90141a564170c08`.

## Search trajectories — VALIDATED

### Kernel Executor

The isolated Kernel ledger contains 23 contiguous rounds, including cold compile
losses, interrupted/failing rounds, source-backed preflight rejections and valid
full runs. The hard cap was 40; the official plateau rule stopped the search at
R23 after the confirmed R19/R20 frontier was followed by three genuinely distinct
misses (R21 packed cross-attention K/V, R22 output-layout attempt, R23 projection
plus residual addmm).

- Selected source commit: `7be2d6b0282f0230b9d888b2412d073ec9964250`.
- Confirmation config commit: `27912ac50009a96dba7a773d56a0d191f1eea477`.
- R19/R20 median: 40.8 s, 1.5123x over the frozen baseline.
- Delivery commit: `e8684f3fa9077d1387de44bbb0521a38ac6b7097`.
- `verify_delivery.py --lossless`: exit 0, `objective_ok=true`, no issues, 50 steps and 100 logical DiT calls preserved.

### Cache Executor

The isolated Cache ledger contains 12 contiguous rounds. It records the rejected
R1 bookkeeping bug and matched EasyCache, TeaCache and TaylorSeer measurements at
conservative, balanced and aggressive budgets. Genuine convergence was reached at
R12, before the hard cap of 20.

- Conservative R2: 46.2 s, 9 hits, low sampled-frame severity.
- Balanced R12: 38.8 s, 17 hits, low sampled-frame severity — selected for integration.
- Aggressive R9: 28.4 s, 29 hits, medium sampled-frame severity — retained as tentative, not integrated.
- Delivery commit: `e7cf11c877a91220af2f2ea2cc5e38000c0765f8`.
- Cache controller direct unittest: 9/9 pass.

Both temporary executor children failed their bounded final handoff. The primary
integrator reconstructed/verifed the deliveries from the already durable ledgers
and continued the Kernel plateau without adding an agent. Details are in
`reports/CHILD-HANDOFFS.json`; no search measurements were discarded.

### PISA — NOT_APPLICABLE

No Attention/PISA child was launched. SANA-Video 2B implements ReLU linear
self-attention with head dimension 112. The archived PISA contract is a piecewise
softmax-attention method, and its backend supports `[32, 64, 96, 128, 160, 192,
224, 256]`, not 112. Adapting it would change the attention algorithm and add a
new kernel shape, outside faithful reproduction. See
`reports/PISA-NOT-APPLICABLE.json`.

## Integration and final H100 measurements — VALIDATED

Integration branch `repro/sana2b-kernel-cache-integrated` composes exact Kernel
R20 with balanced EasyCache R12 at composition commit
`6d9ad0e984c6d236da7be3c3bc0e3d513f23173c`. The only merge conflict was the
runtime fingerprint list; it was resolved by retaining both Kernel and Cache
sources. `py_compile`, four config dry-runs and 9/9 direct unittests passed.
`pytest` is not installed in the validated environment, so it is reported
`NOT_RUN`; the same unittest file was executed directly.

| Prompt | Dense | Integrated | Speedup | Runtime peak | nvidia-smi peak | Cache |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Model-card long prompt | 61.7 s | 29.2 s | 2.1130x | 18,102 MiB | 23,653 MiB | 18 hits / 32 computes |
| Default cat-and-dog baking prompt | 60.5 s | 33.0 s | 1.8333x | 18,088 MiB | 23,639 MiB | 16 hits / 34 computes |

Integrated prompt-1 run:
`runs/20260831-183230-sana_video_2b_h100_integrated_kernel_r20_cache_r12-closeout-integrated-prompt1`,
video SHA-256 `eb05b13437c86c51de02e75bdc8a71b652384316d73c7d92bb0efb40479ef1e3`.

Prompt-2 dense control:
`runs/20260831-182800-sana_video_2b_h100_dense_baseline-closeout-dense-prompt2`,
video SHA-256 `2d91c29834bd4a03608993aaf7555b84a26c4de3d1d35661031a89fd7dc2e0c9`.

Integrated prompt-2 run:
`runs/20260831-183031-sana_video_2b_h100_integrated_kernel_r20_cache_r12-closeout-integrated-prompt2`,
video SHA-256 `da75caee9ecd44e54ae53de1fe2a090f46ef6c2e9aff60bde859e5ad1e5502af`.

All three closeout runs returned the leased UUID to 0 MiB with no compute apps or
residual process. All videos validate as 832x480, 81 frames at 16 fps. Manual
first/middle/last inspection passed for both prompt-matched dense/integrated
pairs: requested subjects and scene semantics remain visible, sampled-frame
geometry is coherent, and no sampled-frame collapse was observed.

## Status taxonomy

### VALIDATED

- Official archived harness recovery plus separately tested current-Codex/cu128 compatibility.
- Model, source, environment and UUID receipts.
- Fresh formal dense baseline on the leased H100.
- Complete 23-round Kernel trajectory through the archived plateau rule.
- Complete 12-round Cache trajectory through genuine convergence.
- Deterministic Kernel and Cache deliveries and the integrated composition.
- Final integrated candidate on exactly two distinct prompts, with prompt-matched dense controls, ffprobe validity and sampled-frame visual checks.

### PARTIAL

- Quality conclusions are bounded to two prompts and first/middle/last sampled frames; they do not establish broad or full-temporal perceptual equivalence.
- The frozen wrapper's `total_s` label says the one-step warmup is excluded, but observed warmup variance leaks into the outer timing. The contract was kept immutable and the caveat is preserved.
- Cache acceptance uses the approved lightweight protocol, not the archived LPIPS/blind-review gate.

### NOT_RUN

- VBench, LPIPS, archived independent validator, full paper table, B200/B300 numbers, a second model, productionization.
- Attention/PISA executor (structurally `NOT_APPLICABLE`, not a failed run).
- `pytest` command (package absent); direct unittest of the same Cache test file passed 9/9.

## Durable acceptance packet

- `INTEGRATION-MANIFEST.json`
- `INTEGRATED-DELIVERY.json`
- `reports/MULTI-PROMPT-SANITY.json`
- `reports/PISA-NOT-APPLICABLE.json`
- `reports/CHILD-HANDOFFS.json`
- `reports/FINAL-REPORT.md`

Large videos, weights and logs remain uncommitted in persistent experiment
storage. No branch was pushed and no PR was opened.
