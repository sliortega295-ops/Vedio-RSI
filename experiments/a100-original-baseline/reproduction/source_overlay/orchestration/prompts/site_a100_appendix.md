## Site transport and workload appendix (A100 reproduction only)

The preceding archived Sol-Agent role, method search loop, correctness gates,
and Master verification duties remain authoritative. This appendix changes
only how this site starts and observes those runs.

### Zero-search information boundary (explicit user requirement)

- Do **not** read or query Codex memory, including
  `/home/lyy/.codex/memories/**`, even if a general memory convention would
  normally suggest a quick pass. Do not use remembered candidate outcomes.
- Do **not** read earlier SolAgent or SolRolloutBench experiment trees, search
  ledgers, reports, candidate code, recipes, timings, or artifacts; do not
  search the web for previous run outcomes. Do not inspect or reuse other
  `output/orchestrated` or `output/experiments` trials in this checkout. If
  recovering state, use only this goal's experiment sequence and worktrees.
- Allowed inputs are this checkout's archived Harness contracts and current
  site adapter, the assigned clean SGLang source snapshot, pinned read-only
  model/dependency assets, the explicitly frozen dense baseline, and new
  evidence produced within this goal. Public source/tool documentation may be
  consulted for implementation details, not previous optimization results.
- If an earlier-trial artifact or historical result is encountered, do not use
  it; record the boundary incident and stop rather than continuing a tainted
  search.

- The coordinator is the current local project checkout. The only GPU target is
  the assigned eight-A100 node, exposed as a pool of eight UUID-checked slots.
  Each inference run uses **one** slot and runs the frozen five prompts
  **sequentially**. Do not infer eight allocated GPUs or distribute the five
  prompts across cards. You may decide whether independent Executors overlap.
- Use the archived `scripts/launch_config.py <toml> --mode sbatch --confirm-submit`
  CLI. For this model its project-local `SOL_SITE_REMOTE=1` shim submits the
  normal run bundle to the authorized node. The node has no Slurm/Codex CLI.
  `python orchestration/site_remote.py status <run_dir>` reports job state;
  `python scripts/collect_run.py <run_dir>` pulls finished outputs and performs
  the archived collection/quality lifecycle. Do not directly execute GPU jobs.
- Weights and dependency overlay are pinned read-only. The editable SGLang
  source is `external/sol_runtime` in each Executor worktree; the coordinator
  dense control uses the clean project-local source snapshot. Do not fetch,
  move, copy, or edit model weights.
- Do not download new packages, model weights, or quality-evaluation weights
  through the local browser, VPN, HTTP clients, package installers (including
  networked dry runs), or remote node. This includes LPIPS dependencies and
  AlexNet weights. Inspect only already available read-only site assets. If
  exact LPIPS cannot run from those assets, record it as BLOCKED; do not
  silently substitute another metric or promote a Cache result on speed alone.
- The independent remote observer was started before the baseline. Raw
  one-second GPU/CPU samples live under project `telemetry/observer/`; remote
  runner events under `telemetry/events.jsonl`; local agent events under
  `telemetry/local_events.jsonl`. Retain job receipts and failed attempts.
- The first GPU job starts a fixed six-hour first-execution budget. If it
  expires, stop launching work, leave results resumable, and report PARTIAL.
  Do not weaken the archived correctness or visual/method gates.

### Coordinator-owned Phase 2 continuation (site transport only)

The Phase 1 six-hour window has expired and its budget, reports, and raw
receipts remain immutable. The user has separately authorized a Phase 2
continuation, but this does not grant an Executor direct GPU or network access.
Only after the coordinator arms a distinct project-local Phase 2 budget and
starts a fresh eight-UUID passive observer may new, explicitly tagged run
bundles be submitted through the same typed broker. The broker and trusted
remote worker enforce a new maximum six-hour window from the first Phase 2
GPU job; do not edit either budget file or bypass the broker. Any LPIPS/AlexNet
dependency provisioning is coordinator-owned outside your Codex session.
Executors and Master must still use the archived quality and independent
verification gates, and must not download packages or weights themselves.
