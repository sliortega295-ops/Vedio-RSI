# SANA-Video 2B executor overlay

This file narrows site-specific parts of the recovered official executor prompt
without replacing its technique scope, search logic, or round budget.

- Run only inside this experiment worktree. Do not edit the coordinator, model
  snapshot, cu128 environment, dependency overlay, kernel staging, lease file,
  or frozen baseline.
- Do not spawn another agent. This worktree has exactly one executor owner.
- Use `python scripts/launch_config.py <config> --mode local`; this cluster run
  does not use Slurm. Every launch must go through the supplied SANA wrapper,
  which takes the shared UUID flock and rechecks foreign compute applications.
- Use the exact frozen 832x480, 81-frame, 16-fps, 50-step, guidance-6, seed-42
  model-card prompt for every screening round. Do not create a five-prompt grid
  per round. The master handles a small multi-prompt sanity check only for the
  near-final winner.
- Do not run VBench, Gemini, an independent validator agent, or a full blind
  review graph. Preserve the output MP4, ffprobe receipt, first/middle/last
  frames, and a concise built-in visual/validity note. Cache may change visual
  output; record the observed tradeoff honestly.
- Kernel has a hard cap of 40 rounds and may stop only at target, cap, or a real
  plateau of roughly 3-4 genuinely different hypotheses without a new best.
  Cache has a hard cap of 20 and may stop early only on genuine convergence of
  the applicable faithful cache search. Do not invent a smaller convenience
  budget.
- One round is one concrete config implemented, launched, evaluated, and gated.
  Record every round, including build failures, runtime failures, OOMs, invalid
  outputs, no-hit cache points, and regressions, in `TRAJECTORY.jsonl` using
  `models/sana_video_2b_h100/trajectory.py`.
- A round record must identify its hypothesis, session/prompt, parent and
  baseline provenance, diff/commit, build, exact launch command, latency, peak
  memory, output/frames, lightweight validity/visual note, decision, and reason.
- The frozen baseline must never be rerun. Candidate comparisons use its exact
  `total_s` and `timing_scope` from the appended frozen-baseline block.
- The final `DELIVERY.json` may contain the real best point and an honest gap or
  convergence report. Do not fabricate three tiers when the measured search did
  not produce three useful points.
