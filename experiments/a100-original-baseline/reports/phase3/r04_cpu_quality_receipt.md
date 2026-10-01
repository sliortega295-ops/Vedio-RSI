# R04 offline original LPIPS backfill (numeric evidence only)

After R05's trusted GPU worker finished at 05:58:32.740086 UTC (exit 0),
the coordinator invoked the **typed, exact-R04** host CPU quality route.
The call did not authorize GPU work, alter a candidate, choose a frontier
point, or execute Agent-owned control code. Preflight checked the current
local R04 eight-file predecessor, remote trusted quality worker SHA
`3088b1fb…`, interpreter SHA `7c9e72ea…`, and three job queues with
0 pending/0 unresolved. The same source had passed 14/14 route unit/negative
tests and `git diff --check` before activation.

Host request ID `c3bca88157c24748897a0191e4b042f7`, request SHA
`cc734eefb64d8a7d3160096511e8a7906e283967e60b0d78fdd90172e3720e73`,
broker PID 1616661, response SHA
`ab3142574b197bb262deba0e1eb22a4a4654c54e0d552fe1e9825083bda386b0`.
Host transport ran 06:02:44.305891–06:03:16.898061 UTC,
32.592179001 s wall; full receipt is
`artifacts/phase3/pre_quality_snapshots/host/R04/20260914T060244.303418+0000_1616661/transport_receipt.json`
SHA `a21d8f14870c74a7f83c63a78c2bee08877be6e26f26d90884d275487155c8c4`.
It snapshots the native local pre-quality JSONs exactly, then SHA-checks
only four small post-quality JSONs when syncing; remote raw run/video/frames
remain remote.

The trusted remote worker ran 06:02:44.617118–06:03:01.295515 UTC,
16.678428871 s wall (preflight 0.732559689 s). Its full original-run
pre/post/pair/CPU/network receipt is
`repo/output/experiments/sana2ba100-cache_ca-0002/worktree/runs/20260914-051802-cache_ca_r04_taylorseer_t0040_o1/phase3_quality_attempts/20260914T060244.617118+0000_2967403/receipt.json`
SHA `87810203501a76103f066e8d3040dcf3970a74358ea6d5fc521693d602b97f09`.
The original archived collector SHA `34d85bf…` (only site timing fields
preserved) ran 13.839173443 s, child CPU lower bounds 324.887613
user-s + 19.696880 system-s. The unmodified original `plan_eval.py`
SHA `a06d9e8f…` ran with `--no-gemini --no-refresh-collection` in
0.118781478 s (child CPU 0.041537 user-s + 0.012461 system-s), so the
collector was refreshed exactly once. Network guard loaded in three
subprocess PIDs, recorded blocked network attempts `[]`; no network
fetch or GPU was used. During its 16.678428871 s same-host monotonic
interval the independent 1 Hz observer had 17 starts, full covered
duration, no error/incomplete sample, and 0 >100 MiB/app/util-equivalent
seconds across **all eight UUIDs**. The quality route was CPU-only and
did not overlap R05's measured requests or worker wall.

The archived pair selector had 80 available aligned candidate/dense pairs
but scored **41**, with per-prompt counts 7/6/6/6/16. Exact selected
indices and every per-frame score are in the remote receipt. Per-prompt
mean LPIPS is p01 0.203572, p02 0.483573, p03 0.352739, p04 0.167635,
p05 0.523164; original judge aggregate mean is **0.385836782615** and
max **0.537195324898** (p05). These are the original `lpips==0.1.4`
judge and fixed AlexNet/linear weights, not pyiqa or a substituted metric.
The post benchmark SHA
`8d2c1a62c5a458bdcb552eb5ff43acbdd7ebaf5fb28aee01449a49e5e11db14a`
still has five request seconds 76.68+76.39+76.43+76.36+76.41 =
382.27 s, exact frozen timing scope, five videos, and 80 extracted frames.
Post quality SHA `ee22905f…`, collection `55b80b5b…`, and assessment
`c91f07ac…` are chained from the host/remote pre-snapshots.

This is **LPIPS numeric/transport PASS only**. Original collector quality
remains `blocked_quality` with `nvidia_gemini:api_key_missing`; standalone
`plan_eval` still reports `baseline_total_s=null`, `speedup=null`, and
`tier=null`. Cache9's R04 visual FAIL file SHA `8c94cf3a…` was generated
without any R04 image attachment and is externally `UNSUPPORTED/REVIEW`,
not a real pixel-visible verdict. No Cache DELIVERY, independent Master
visual gate, native Cache verifier, or integration follows from this receipt.
Any second R04 reassess must first pin this exact posthash/receipt as a
new approved predecessor; the initial route is one-shot by design.
