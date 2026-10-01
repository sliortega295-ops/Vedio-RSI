# Original Sol-Agent A100 baseline: acceptance checklist

This is the primary conversation's review contract, not an Executor search hint.
The archived orchestration-lite Master/Executor workflow is the baseline being
observed; site transport, the pinned SANA-Video 2B workload, and the fixed
six-hour GPU budget are explicit adaptations.

- Freeze one real dense five-prompt baseline with a pinned model/source,
  unchanged 50-step/81-frame/832x480/seed-42 inference contract, raw video
  provenance, and one authoritative request-time timing scope.
- Preserve the clean sequence's Master and Executor prompts, session attempts,
  candidate configs, failed submissions, raw runs, collections, gate decisions,
  DELIVERY files, verification results, and integration attempt. Never promote
  a result from the contaminated sequence 0001.
- Keep the zero-history boundary honest: record any visible metadata exposure,
  external skill instructions, restart, or Master intervention; do not claim
  stronger isolation than the trace shows.
- For each candidate, distinguish submitted/started/GPU-finished/collected/
  quality-gated/independently verified/integrated. A fast raw benchmark is not
  a validated frontier point.
- Apply the archived lossless structural and method audit to Kernel. Apply the
  archived aligned LPIPS plus visual gate to Cache. Missing evaluation assets
  are BLOCKED, not a substitute metric or a quality PASS.
- Validate identical timing scope before computing speedup. Keep profiler,
  warmup, build/load, video export, queue, and candidate-decision wall time
  separate from the frozen request-time denominator.
- Report eight fixed GPU UUIDs at 1 Hz: allocation GPU-seconds, observed
  utilization-weighted seconds, memory/power, concurrent jobs, idle windows,
  and an explicit distinction between eight available cards and one card per
  video. Do not call 1 Hz NVML utilization precise SM occupancy.
- Report local Codex tool/attempt windows and remote CPU/process phase windows
  from their respective monotonic clocks. Do not subtract local and remote
  monotonic timestamps or assume NVML host PIDs identify container `/proc`
  processes.
- Attribute control-plane and GPU time separately to archived Harness,
  site adapter, agent environment/skills, safety recovery, and actual model
  inference where evidence permits; otherwise label attribution unknown.
- End with either an archived independently verified integrated delivery or
  a reproducible PARTIAL at the non-resettable GPU deadline, with every
  unpassed gate and unfinished stage explicit. No silent deadline extension.

