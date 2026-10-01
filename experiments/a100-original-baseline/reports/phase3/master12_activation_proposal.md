# Master12 exact Cache acceptance transport — proposed, not activated

UTC 2026-09-14 10:05. The native Cache11 `DELIVERY.json` SHA is
`0940aca6c4e90502b808f5d3441f49a7b3d0d1ac1c2f10a4170e6e4ab2920e85`;
its full frontier is exactly R05 and R08. Master11 was stopped at
10:02:09.879302Z after a native local verifier false-positive and unsupported
visual claim; no integrated GPU run was submitted.

## Proposed Master12 Codex adapter delta

For **only** `sana2ba100-master-0002`, attempt `12`, before the Codex child
starts:

1. Pin/import `repo/tools/symposium/site_phase3_master_cache_images.py` SHA
   `9defded11bed4b1a1f8c36b00b200e97c5f02428f18c389c95df4bf113e3e204`.
   Its `inputs(name, attempt)` must return exactly the approved image set;
   failed validation aborts the Master session, with no fallback image route.
2. Set the child environment's `PLAN_EVAL_PYTHON` to the fixed coordinator
   `repo/orchestration/site_reverify_python.py`, only after the two exact
   `site_reverify_transport.assert_approved(run)` checks succeed. No other
   Master/Executor attempt inherits this environment override.
3. Include the 20 `--image` arguments and `PLAN_EVAL_PYTHON` shim path in the
   `agent_api_begin`/receipt transport event. The shim is the existing exact
   archived `search/plan_eval.py` argv parser; it sends a typed coordinator
   `reassess_quality` request, never arbitrary shell or an Agent-owned worker.

The prepared image attachment at
`artifacts/phase3/cache_image_transport/attachments/sana2ba100-master-0002-attempt-12-cache.json`
has SHA `9dfbbdf11bbd632bfe29a6fc07d8478172277d73f3138d13f74b3a1c50364915`,
exactly 20 images/48,469,025 bytes: R05 and R08 × five prompts × overview/detail.
It remains `prepared`; no Master has received these images.

## Proposed CPU reverify activation order

1. Review the staged remote worker (`repo/orchestration/site_reverify_worker.py`)
   SHA `b57cfd74f0eea24abea5584e20e0ecc20c19ba5b3ae364d0e717fea435338dca`,
   host route (`repo/orchestration/site_reverify_transport.py`) SHA
   `6c87150dcfd1dda0e1e14a5bca52bfcd62ea3b652a92bc0c05bf321a6a651bbc`,
   and immutable first-judge receipts. These stage only R05/R08 and retain
   raw `nvidia_gemini:api_key_missing` blockers. Host and remote metadata
   have separately pinned SHA because site provenance differs by host.
2. Use the staged pre-call guard to recoverably move each native Master11
   `reverify_verdict.json` into its exact run's versioned archive. The old
   SHA values are R05 `21a7bd76b0e9a9300a077c7dbaf8b0b6895142bf286544fa9bb5f4553dfebb64`
   and R08 `f1ed56e8c3cc11b4486d0d7fa2c3f107d4b223ee45cda16f29abb5f6507b34ca`.
   Pin the resulting guard receipt SHA/path into the host approval map; fixed
   output paths must be absent before Master12 starts.
3. Pin the final trusted remote worker SHA in the host route, sync only that
   coordinator-owned file to the remote project, verify before/after SHA and
   retain its old version. Require all typed queues pending=0/unresolved=0,
   no active GPU or quality assessor, then scoped-restart only the host job
   broker and verify new PID/heartbeat/source SHA. No GPU authority is added.
4. Change only the Master attachment `status` prepared→approved, rehash and
   run the exact non-override 20-image CLI dry-run. Apply the attempt12-only
   adapter delta, scoped-restart the goal broker while no session is active,
   and run its tests/dry-run.
5. Append dated factual correction to the same archived Master goal: first
   local objective boolean was not an independent LPIPS run (host lacks
   torch/torchvision/lpips; quality hashes unchanged); 80/80 visual claim had
   no image input. Point to the approved typed CPU bridge and exact sampled
   pixels, without supplying a visual verdict or candidate instruction. Resume
   the same Master identity as attempt12 via native goal broker.

Accept Master12 only with `agent_api_begin` showing all 20 image paths and the
shim, real remote worker/host receipt chain, original collector+LPIPS exits0,
fresh `reverify_verdict.json` hashes, `_plan_eval_rc=0`, and prompt-specific
Master observations. The archived verifier source and objective rules remain
unchanged. No composed GPU submit until those gates are verified. Phase2
new-submit cutoff is 10:32:50.441475703Z; hard cap 10:53:50.441475703Z.
