# R06 original LPIPS CPU route — staged, not executed

At 2026-09-14 07:06 UTC, after the original Cache Executor's native local
`plan_eval --baseline-frames` completed, the exact real R06 was pinned for a
single future offline original-judge assessment. This does **not** approve a
frontier point, restart the broker, run the judge, or grant GPU authority.

Run: `repo/output/experiments/sana2ba100-cache_ca-0002/worktree/runs/20260914-063907-cache_ca_r06_easycache_t010_w30`, trusted remote worker PID 2982383
exit 0 at 06:55:57.464777Z. Five 832×480/81/16 videos/80 aligned frames;
same five prompt hashes/seed/steps/fixed revision/timing scope, one assigned
GPU UUID0. Local benchmark `total_s=480.37` equals
96.39+95.99+95.98+95.98+96.03. Remote raw runner benchmark SHA
`2dfedb8ad1a2fb4ed68729a8fe3c42adcc1cb66309d5c3c69e5d5a98a5787972`;
remote/local run_config SHA
`6f32cda5b35e547e6d66e39b1b46e892526940aaa5f1619fa4b44d4de7e02880`.
Remote/local original 80-frame content digest using the worker's ordered
`sha256(frame)+newline` algorithm is
`4c9f96ea9e749688ea335e705731b3b41c7460313902edae800824fdd670717c`.
Five MP4 SHA values match the export manifest; see
`reports/phase3/r06_execution_observer.md` for worker/observer stage evidence.

Pinned current local eight-file preassessment state:

| Relative path | SHA256 |
| --- | --- |
| `launch.sh` | `e214d22a26dab33c3c1e9de79e9089ad5e40bb27a4b06203052527af43c702b7` |
| `manifest.resolved.toml` | `fce503fabbb219c9cab0df441b6555163eb3e66c115e9f7119febd46d9530b9f` |
| `metadata.json` | `268d587870279c18bdf7a71682122315eb1370b12ea3d0d593368a00cd8ceae3` |
| `outputs/benchmark.json` | `f31a3cf05715bda16b4576931e33b95b43ec07fcbcea46bb2fff39efb9b0f206` |
| `outputs/run_config.json` | `6f32cda5b35e547e6d66e39b1b46e892526940aaa5f1619fa4b44d4de7e02880` |
| `outputs/quality.json` | `0627682df8bcefc782006e52ea6d864cdccd40a0c276f1ebe55747726defc570` |
| `outputs/collection.json` | `0ae49de1f193b6400093542cd25a5da49ad0e9f43e949e4d4b9a6b0533bacc88` |
| `assess_verdict.json` | `15ad43895f67e2693ea2b327ced9f39b232d2eff386b07729944ad83620f12b1` |

The local native assessment currently reports LPIPS dependencies missing and
raw Gemini `tool_missing`, not a quality verdict. The staged transport would
run the archived `collect_run.py` SHA34d85… and original `lpips_judge.py`
SHAf68f… on the already authorized remote CPU interpreter with fixed official
LPIPS wheel/linear/AlexNet SHA, `CUDA_VISIBLE_DEVICES=''`, a socket-deny
sitecustomize guard, and project scratch. It snapshots source bytes and hashes
before/after, checks original selected aligned pair prompt distribution, and
syncs only small JSON/receipt files. It may update the real run's quality JSON
as the archived collector requires; it cannot alter the frozen dense benchmark,
re-run GPU generation, choose a candidate, or claim visual/Master PASS. The
generic collector's Gemini blocker remains visible; lite-vs-graph contract
conflict is separately reported.

Local-only staged source SHA:

- `repo/orchestration/site_quality_worker.py`:
  `4e112dcbfb228489342f8200e5df916b535a0bd0e8dc2784e2f482618c8df861`
- `repo/orchestration/site_quality_transport.py`:
  `3cae845686638bb754d4f399f6013c6f9a563bfb8c83b677267d3260d20a14e6`
- `repo/tests/test_site_quality_route.py`:
  `24136c2b8694926044a67aae53e86add33c9ce7053350c96fa050c505f405632`

Combined local negative/identity tests: 33/33 PASS in 0.930s, including exact
R06 local bundle+80-frame digest and a bad benchmark SHA rejection;
`git diff --check` PASS. At this staging point the remote trusted worker remains
the R05 version SHA525fd4dd…, and the active host job broker PID1669689
pre-dates these edits. The new route cannot run through that active broker.
Before any later R06 CPU assessment: recheck all exact local/remote input hashes;
require all typed queues pending/unresolved 0 and no active assess; wait for any
subsequent candidate GPU worker to finish to avoid CPU/I/O interference; sync
only the hash-pinned trusted coordinator worker with prior-byte backup; scoped
restart of only the host job broker; verify new PID/heartbeat/trusted SHA;
then one typed CPU request and independent post-state receipt review. Any
intervening Agent mutation fails closed and requires new review.
