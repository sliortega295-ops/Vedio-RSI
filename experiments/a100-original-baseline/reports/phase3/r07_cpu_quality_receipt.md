# R07 original offline LPIPS: numeric evidence only

On 2026-09-14, one coordinator-triggered, typed `assess_quality` request ran
the archived collector and LPIPS judge against the real seq0002 Cache R07.
The exact [R07 staging receipt](r07_quality_route_staging.md) pins its five
videos, 80 aligned frames, run/benchmark predecessor, original judge, frozen
dense reference, and offline dependencies. This CPU operation did **not**
launch a candidate, use GPU, decide the frontier, or supply R07 pixels to
Cache9/Master11. Cache9 attempt9 had only R02 actual `--image` inputs, so its
R07 visual PASS remains externally **UNSUPPORTED**. Original Gemini remains
blocked and no Cache DELIVERY or Master independent R07 review follows from
this numeric result.

## Trusted transport and predecessor chain

Immediately before activation, local `validate_local(R07)` matched all eight
pinned files. The remote raw benchmark and run-config SHA, five MP4 SHA, all
80 candidate-frame content digest, and fixed collector/plan-evaluator/judge,
wheel, linear/AlexNet weight, guard, and interpreter SHA matched the staging
receipt. All eight fixed A100 UUIDs had no compute application in the
preassessment `nvidia-smi` query. The three fixed broker queues had pending=0
and unresolved `.processing`=0; the old host site-job broker had no child.

The trusted remote quality worker's predecessor SHA
`4e112dcbfb228489342f8200e5df916b535a0bd0e8dc2784e2f482618c8df861`
was copied to this project's remote
`artifacts/phase3/trusted_worker_pre_r07/site_quality_worker.py` and rehashed
equal. The new single-file staged source SHA
`374b56dd1a47497a8c2879c5ffa4a1ddc544b4a68342c80c1cfb7a66b017693c`
was sent as one checked 7,629-byte archive (SHA
`a37ce46d06a6afca26f149a971ddb69eb26e4d615c4e5db6889baf4a545b23cd`),
rehash-checked remotely, then moved into the exact trusted coordinator
`repo/orchestration/site_quality_worker.py`; final remote SHA matched. Only
the host site-job broker was scoped-restarted, PID `1756127` → `1829680`,
new start `07:58:37.600892Z`; its heartbeat and loaded coordinator source
were checked. The goal broker, Master11, Cache9, GPU observer, and any
candidate source were not restarted or modified.

The one host request ID `dc38c92c0df1439fb9d2b3812ca993a0` was written to
the exact Cache seq0002 typed queue. Its canonical request SHA is
`b840c7ae995eb59e4aa14abf61464261c8ac9f469bc49890494247d2f24188a8`,
canonical response SHA
`22a3e7cdab3892d65e15d6792c34d09e556047524568a5283228c8e08d91e37d`;
full request/response file SHA are respectively
`c36771007134d0acebec044ab2036778225f4196fcaed0dbf3bc09b4345cdeb8`
and `a358805bbd9555c2ee047edb6c13a0815e0f5d2adc5e06566a1eca8cd0776b10`.
The coordinator-owned broker client selected this one exact Executor queue
for the approved site CPU bridge; it did **not** execute Agent-owned worktree
client/control code. The host broker independently validated queue, exact
run path, launch/manifest/metadata SHA, and allowed operation. No arbitrary
shell command was accepted from the Agent.

Host transport receipt (retained locally: `../../artifacts/phase3/pre_quality_snapshots/host/R07/20260914T080117.986347+0000_1829680/transport_receipt.json`)
SHA `27dd9d1efcb3269728b7cec1166b52c2ee10cbaf2a10e66536b2d6e00fad0673`
records one remote exit 0. The original remote
quality receipt (retained locally: `../../repo/output/experiments/sana2ba100-cache_ca-0002/worktree/runs/20260914-072228-cache_ca_r07_teacache_t00087/phase3_quality_attempts/20260914T080119.530175+0000_3026250/receipt.json`)
SHA `f275f3740a99a2367d1a2d7dae04e6e19b3c9c5408fea196072cfbc33c521356`
includes the full selected pair indexes, scores, source/dependency hashes,
CPU/stage clocks, guard PIDs, and remote pre/post snapshots. Only small
JSON/receipts were pulled. Both host and remote prior quality bytes were
snapshotted before archived collection refreshed the authentic run.

The host prior benchmark/quality SHA were
`ee4d80b76c56cfa9a2eab2b5921949204411a5dc7e5ab149f9b92a09514524e4`
and `3125086633bd8454bcc4072b921201fd7ab47537009ec85144b483154637315c`;
post SHA are
`b97ee19b606f08a98e1e0d2e8d2c0704cd9a3c59efd5eba0c8f34cb3e5a66877`
and `e6645c573bb6e33c9766ba25e4a271cab0cf388badddc5c0e064b4a9b6bbb88a`.
Post collection SHA is
`79e5dbcd7f44a710ad3366f73e6051e43f79323799e8446083c93ab834678e51`;
post assess SHA
`40269e186dea3550d87724595b4321cc1cfab240ea3bb36573517c255af813ee`.
Remote raw benchmark predecessor SHA `3f582161...` became the same post
benchmark SHA as host. The frozen five request times still sum to **480.18 s**,
timing scope unchanged, five MP4 SHA and 80-frame digest unchanged. The
first-evaluation SHA pins now deliberately reject any second R07 assessment;
future independent reassessment requires an explicitly approved post-hash
and receipt chain.

## Original judge output and measured time

The original collector selected **42 of 80** aligned pairs via its
stratified-plus-worst rule, p01–p05 distribution **7/16/6/6/7**. Its LPIPS
result is mean `0.02434319463957633`, median `0.015329861082136631`,
max `0.04971546307206154` at p02/f_032. Per-prompt means are
`0.0146384316/0.0407439040/0.0151733298/0.0142757039/0.0130497840`.
Original `quality.json` remains `status=blocked_quality` with
`promotion_blockers=[nvidia_gemini:api_key_missing]` even though its LPIPS
judge is `complete`. `assess_verdict.json` still has `baseline_total_s=null`,
`speedup=null`, and `tier=null`; this site route does not fill those fields or
replace original Cache/Master decisions. No LPIPS-only quality PASS is claimed.

The remote quality worker ran `08:01:19.530175–08:01:37.547187Z`, same-host
monotonic wall **18.017044723 s**: preflight `0.722726402 s`, archived
collection `08:01:20.328681–08:01:35.351392Z`/**15.022718767 s**, and
original no-refresh plan assessment
`08:01:35.437716–08:01:35.659633Z`/**0.221919898 s**. Collection child
rusage lower bounds are **353.941729 user + 19.486150 system CPU-s**;
quality-worker process CPU is **9.728076343 s**. Host typed transport wall
was **20.954193271 s** (`08:01:17.987809–08:01:38.941989Z`). Cross-host
monotonic clocks were not subtracted. Three child PIDs reported network guard
loaded (`3026260/3026338/3026609`), and
`blocked_network_attempts=[]`; CUDA was hidden by the trusted worker.

Independent remote 1 Hz observer selection at `08:03:00.909Z` read 22 rows
around the exact worker monotonic interval
`[1881229753811643,1881247770856366]`, selection SHA
`74d133ffbf3d69f813e020bb6ef58f3ad494128ecf24cbc3527d0f5a361aa344`.
Its 18 in-window samples cover all **18.017044723 s**, 0 unobserved seconds,
0 error/incomplete-UUID rows, max sample-start gap `1.000439957 s`. Each of
the eight UUIDs has **0 sampled app-present, >100 MiB, and utilization-
equivalent seconds**. Exact worker-lineage `/proc` sampled CPU lower bounds
were **353.50 user + 15.74 system CPU-s**, separate from child rusage; sampled
RSS peak 1,238,147,072 bytes. The host broker's next audit entry after R07's
07:42 status was this single CPU assess; there was no candidate submit in
the judge interval. Sampled zero GPU use cannot prove an absence of every
sub-second kernel, but the route grants no GPU and hid CUDA.

The archived lite Cache visual gate, Master independent visual gate, genuine
Cache DELIVERY, native verifier (including its repeat LPIPS transport), and
integration remain pending. This receipt is **numeric LPIPS evidence only**.
