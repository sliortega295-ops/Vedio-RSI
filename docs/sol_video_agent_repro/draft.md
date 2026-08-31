# Sol-Video-Agent SANA-Video 2B / H100 reproduction draft

## Task contract

- **Task:** recover the archived Sol-Agent harness and reproduce a fresh SANA-Video 2B optimization search trajectory on one H100.
- **Objective:** start from one frozen dense run, let exactly one Kernel executor and one Cache executor search from that identical baseline, retain all attempted rounds, integrate accepted winners, and stop after one integrated smoke run.
- **Inputs:** harness commit `d2c6407cc9b9133f3fff49fe4b561f14980d3f8b`; runtime commit `b0b7eb4d0a7f1f46118a356485f4523cf52e96dd`; model revision `db5f398b13ca086d09a50ce156c20527773841b1`; one fixed H100 UUID selected after a fresh ownership check.
- **Outputs:** archival and compatibility commits/patches, `BASELINE.json`, per-round Kernel and Cache ledgers, both `DELIVERY.json` files, PISA `NOT_APPLICABLE` evidence, `INTEGRATED-DELIVERY.json`, videos/frames/benchmarks, and a final VALIDATED/PARTIAL/NOT_RUN report.
- **Correctness:** the workload stays at 832x480, 81 frames, 16 fps, 50 steps, guidance 6, seed 42, the model-card long positive/negative prompts, and positive-prompt suffix `motion score: 30.`; VAE remains fp32 and transformer/text encoder bf16. Kernel candidates preserve logical model work. Cache candidates may skip work only when their lightweight output-validity and visual check passes.
- **Search budgets:** Kernel hard cap 40 with only the archived target or roughly 3-4 genuinely different no-new-best hypotheses as early-stop; Cache hard cap 20 with archived genuine-convergence behavior. No smaller convenience budget is allowed.
- **Allowed implementation:** Python, TOML, shell launch manifests, current `codex-cli`, archived harness primitives, the existing CUDA 12.8 environment, and SANA-specific import compatibility. No CUDA 13 rebuild, B200/B300 matching, VBench, PISA adapter, second model, external validator agent, push, or PR.
- **GPU constraint:** every run uses one frozen physical H100 UUID and the same persistent cooperative lock. The wrapper must check live compute processes after taking the lock and before launch; it must never signal or displace foreign work.
- **Validation commands:** focused Symposium unit tests; archived orchestration dry-runs; CUDA 12.8 import/registry/JIT smoke; config-launch dry-run; ffprobe checks on every retained video; deterministic delivery verification where compatible with the SANA 2B adapter.
- **Evaluation command:** `python scripts/launch_config.py <config> --mode local` inside the model experiment, through the UUID-locking SANA wrapper. Authoritative latency is the warm post-build `generate` interval and peak memory is sampled for that UUID during the process.
- **Promotion:** a candidate must have a committed/diff-identifiable implementation, a real successful GPU run, correct workload receipt, valid MP4/frames, truthful timing/memory evidence, and an explicit accept/reject reason. Integration requires both component deliveries to pass the same checks and a real combined smoke run.

## Current baseline and provenance

- The formal dense baseline is `NOT_RUN`; the earlier public Diffusers result is reference-only.
- The model snapshot is already complete: 20 files, five safetensors, 14,002,562,288 bytes, with the exact requested revision.
- The CUDA 12.8 environment has passed an import plus BF16 RMSNorm JIT gate, but has not loaded the full model or generated a formal baseline.
- GPU 0-5 were foreign-occupied at the first fresh check. GPU 6 (`GPU-83ed65f8-62e5-2a01-3471-8bfc752971d3`) and GPU 7 were idle; no UUID is considered leased until the launch-time recheck and lease receipt.

## Pre-work state to preserve

- Harness branch: `repro/sana-video-2b-full-exploration`, exact HEAD `d2c6407...`.
- Pre-existing uncommitted transport candidate:
  - `tools/symposium/codex_goal_session.py` SHA-256 `4d48ee619f8c2cf30cab2d1dcb61055ef70a6a019975d796180349b53747213e`.
  - `tests/test_codex_goal_session_exec.py` SHA-256 `363c19edaf2cad8a9e5fde8aa2dfae72acc21c5b47fc19cd31e6556513d187e9`.
  - Combined tracked diff SHA-256 `4823b00d7b8fe111a021f7dbe8606b81534c156477d33249c6887991c2ffa442`.
- Pre-existing CUDA 12.8 candidate worktree remains based on exact runtime HEAD `b0b7eb4d...`; its tracked diff SHA-256 is `a50e3068bd2251ba86d2bb35fe74a73a98fb2b04972631e568a44a383dbf7443` plus untracked `registry_sana.py` SHA-256 `553adb19a912672c9e8d81eed2f21c51e01337fcaa4dcca26278cd0f496c7673`.
- The old `20260827-official-repro/video-infra-rsi` group WIP is out of scope and must remain untouched.

## Risks and unknowns

1. The archived harness targets a private SANA 5B profile, not the public SANA-Video 2B runtime. A new model contract and wrapper are required without changing the archived technique scopes.
2. `codex_auto_run.py` is absent. The candidate `codex exec --json` transport must be tested against the current CLI contract and fail closed on process/thread identity errors.
3. The existing CUDA 12.8 changes narrow eager imports. They must be proven opt-in and byte-separable from search/model changes.
4. Shared H100 availability can change between checks. A fixed UUID, cooperative lock, and launch-time foreign-process gate are required for every round.
5. Kernel and Cache agents may propose overlapping runtime edits. They start independently, and the master integrates only explicitly compatible winners after reviewing actual diffs.
6. Exact original NVIDIA private conversation traces are unavailable. The deliverable is a new trajectory under the recovered official prompts, contracts, and budgets, not a claim of transcript identity.

## Ranked implementation directions

1. Accept or repair the small exec-JSON session transport after focused fake-CLI lifecycle tests and static current-CLI option checks.
2. Preserve the runtime authority untouched; commit the SANA-only CUDA 12.8 import adapter on its own branch and export `ENV_COMPAT_CU128.patch`.
3. Add a SANA-Video 2B model contract, local locked wrapper, dense config, receipts, and tests to the harness branch.
4. Run and freeze the formal dense baseline before materializing executor worktrees.
5. Create exactly two executor worktrees from the same frozen baseline, then dispatch the Kernel and Cache child agents with the archived scopes plus the SANA 2B execution contract.
6. Integrate only delivered winners, rerun a combined smoke measurement, and preserve all rejected rounds in the trajectory ledger.

## Evidence required

- Source, model, environment, GPU, command, config, and patch fingerprints.
- Red/green unit evidence for transport behavior and characterization/green evidence for the opt-in import adapter.
- A real baseline MP4 with ffprobe receipt, frames, latency, and peak-memory receipt.
- One JSONL row per executor round, including failures and no-improvement decisions.
- Delivery and integrated-delivery schemas tied to actual run directories and implementation identities.
