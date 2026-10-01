# R04 exact offline LPIPS route: staged, not executed

At 2026-09-14 05:43 UTC, the coordinator staged a site-only extension of
the already-tested typed CPU `assess_quality` route. It admits only the real
seq0002 Cache R04 run `20260914-051802-cache_ca_r04_taylorseer_t0040_o1`.
It does not choose a candidate, submit GPU work, modify archived
`collect_run.py`/`plan_eval.py`/`lpips_judge.py`, or decide Cache quality.

The remote pre-collector benchmark SHA is
`cb36ffc54c66296eabee6a7edc482ceb26101113161d644fc1ae8622feb9bc78`;
its five request values sum to 382.27 s. Run config SHA is
`1a3f75560f3c178358e3cb1859b893120e66b510907fd9275400745e5149dfd1`,
and 80 aligned local frames have SHA-line digest
`f28e2c0f3e63df316b354afd0d318172730024ff09b49fec100b336d1ddefa39`.
At 05:43 the remote trusted worker was still the prior R03 SHA
`2621e434f3b4b410996a7b393cf9d8f53102f03e8252de3330f311401d64328b`.

The current local predecessor was produced by the native Cache9
`plan_eval --no-gemini` at 05:36:00–05:36:14, after its native collect.
Exact hashes pinned by the transport:

| Run file | Local predecessor SHA256 |
| --- | --- |
| launch.sh | `4795bcb36323aadcab8b54b176ab2a969441ccef470fd80181e02e5360349e0b` |
| manifest.resolved.toml | `94091ab4de006122f1616c7cf93872c6bbf9dd65fb40f45100c81fdd62380246` |
| metadata.json | `998c906d5a6e26ce09ed6b06ce99f705553d3e60f92c1697759330f019d648d6` |
| outputs/benchmark.json | `5a2eccb0b9aa49c657689793b545642ba783a1b798019c2c6ac3ebc40b613cc4` |
| outputs/run_config.json | `1a3f75560f3c178358e3cb1859b893120e66b510907fd9275400745e5149dfd1` |
| outputs/quality.json | `71580feef142539a8e2fbecb7726518a2574b09531dcf337b536f860edd5f17e` |
| outputs/collection.json | `599d475eb6b801b1f1f1eac54041a7ecd5edf7ac410855db60d20e28f9bb8614` |
| assess_verdict.json | `4a9cebe56988e64358428d0ef777d740eb8ece9fdd30f3a02fc21a31d446c94d` |

Only an exact prehash match will permit evaluation; a concurrent Agent
rewrite must fail closed and require a newly reviewed predecessor. The
remote worker also checks exact raw SHA, 5 videos, 80 frames, fixed
dependency/source SHA, offline socket guard, and CUDA-hidden environment;
it snapshots pre/post JSON and records the original collector's selected
pair indices and process CPU/wall. R01/R02/R03 repeats remain rejected
until a separate explicit posthash/receipt chain is reviewed.

Staged source SHA256s: `site_quality_worker.py`
`3088b1fb31530293fc48fd8c249a2b7ce1bb9132e618bd65ed843579069a6495`;
`site_quality_transport.py`
`119d8345b011891b723369736ed6e643e532910a1b2a9c62b5c4456b4f5c7059`;
`tests/test_site_quality_route.py`
`8cd5e4b8d891cf9fdadc2e574135ded00dfa3d2a504e2e5d313afaecdb353ed9`;
one-member transfer helper `sync_r04_quality_worker.py`
`55affd8712f03b455809b8f2be7b9a9a9702f3c3fe82240b75066227b2ace0f8`.
The helper's read-only dry-run at 05:43:08 confirmed remote old SHA
and exact project-local backup target; no transfer yet. `git diff --check`
and route tests 14/14 PASS after correcting one staging-hash transcription
error. That initial 1/14 failure is a CPU preflight failure, not a judge run.

Cache9's R04 `visual_verdict.json` at this stage SHA
`8c94cf3abf7be9ae4ea8bcc8886fdc2053236a92e1e0ff067eef9a8ea63b324b`
claims a visual FAIL, but the active Codex session attached only R02
images. R04 `montage`/`identify` did not deliver pixels into the model.
The visual conclusion is therefore `UNSUPPORTED/REVIEW` externally, just
as R03's earlier montage-only PASS was. Exact LPIPS will be numerical
evidence, not an acceptance of either visual verdict. The archived
collector's separate Gemini blocker remains visible.

## Trusted transport activation, still no R04 judge run

After independent coordinator review, at 05:46:38 UTC the three typed job
queues had 0 pending `.json`, 0 unresolved `.processing`, and broker PID
1578907 had no child. The read-only sync dry-run at 05:46:46 again saw the
old remote worker SHA. The exact one-member transfer ran
05:47:00.080853–05:47:04.016590 UTC: archive SHA
`1f6e9c8dce4a6dc4174fa5e027b98bb57164962777a3c71524dff93d05775181`,
7064 bytes, remote `received_verified`, old worker backed up under this
project at `artifacts/phase3/trusted_worker_pre_r04/site_quality_worker.py`
with SHA `2621e434…`, and remote replacement independently SHA
`3088b1fb…`. No candidate run/code/model file was transferred.

With the same 0/0 queue and no-child guard, the coordinator stopped only
the old host job broker PID 1578907 and started PID 1616661 at
05:47:27.551459 UTC. Its fresh heartbeat at 05:47:35.556257 matched
the PID/start time. Host `trusted_remote_worker()` checked the new remote
worker SHA and fixed interpreter SHA `7c9e72ea…`; local R04 exact
predecessor validation passed; route negative tests remained 14/14 PASS.
Master11, Cache9, their goal broker, and the R05 remote GPU worker were
not restarted or canceled. R04 `assess_quality` remains **NOT_RUN** until
R05's measured requests are finished, avoiding a repeat CPU-evaluation
overlap with a candidate GPU run. This wait is a site quality/transport
dependency, not a native candidate decision.
