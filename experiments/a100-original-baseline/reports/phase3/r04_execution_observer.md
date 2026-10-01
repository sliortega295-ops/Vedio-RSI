# R04 TaylorSeer execution and resource receipt (quality pending)

The original seq0002 Cache Executor autonomously submitted
`20260914-051802-cache_ca_r04_taylorseer_t0040_o1` with first-order
TaylorSeer, threshold 0.04 and a one-reuse cap. This was a calibration
hypothesis, not a gate result. Trusted remote worker PID 2932806 started
05:19:28.562319 UTC, finished 05:33:02.751717 UTC, exit 0,
duration 814.135550856 s, no Phase 2 budget timeout. R03 had finished
before R04 admission; there was no GPU-run overlap. A transient SSH
gateway `Connection closed ...:2222` exit255 in the independent auditor's
query was transport-only: trusted status succeeded again at 05:28:58 and
05:32:38 while the worker was alive, then returned the exit-0 final state.

## Raw five-prompt timing and structure

Remote pre-collect `outputs/benchmark.json` SHA256
`cb36ffc54c66296eabee6a7edc482ceb26101113161d644fc1ae8622feb9bc78`.
The fixed five request seconds **76.68/76.39/76.43/76.36/76.41** sum
to **382.27 s**. Source-reported denoise seconds are
70.8855/70.6030/70.6304/70.5670/70.6190; decode seconds are
5.0894/5.0814/5.0919/5.0909/5.0911. Five separate prompt process
wall times are 163.7371/162.6552/159.1940/160.4388/162.5118 s.
The benchmark explicitly excludes repeated model load, internal one-step
warmup, and video export from `total_s`; the larger process walls are a
site wrapper cost and must not be silently added to the frozen objective.
All five prompts used GPU0 UUID
`GPU-cad55364-e014-c35f-337e-db899afd1c6a`, one video per process,
world size 1, 832×480, 81 frames, 16 fps, 50 steps, seed 42. The
timing scope exactly matches the frozen dense and R03 request-only scope.
Five prompt logs each contain 50 `TS step=0..49` decisions with exactly
18 `run=0` first-branch reuses (90/250 logged steps overall). The source
copies a branch-0 decision to the second CFG branch; that second-branch
count is not separately instrumented. No quality judgment follows from
these activation logs.

Remote launch SHA256 `4795bcb36323aadcab8b54b176ab2a969441ccef470fd80181e02e5360349e0b`,
manifest `94091ab4de006122f1616c7cf93872c6bbf9dd65fb40f45100c81fdd62380246`,
post-worker metadata `ef511d7d46a8f33bf19d681f67f2ced9f93a2329c015010c19e5cd1e4abf2879`,
run_config `1a3f75560f3c178358e3cb1859b893120e66b510907fd9275400745e5149dfd1`.
Remote video SHA256s p01–p05:
`893333cfaa6b2665722ed7bfb03c49acb1e72f32f2266a11b6906773211ea50c`,
`8ce063dbab7d0c620e9c963adfeb2e6951cf3cf0fdfed2fd05223723a02ef24d`,
`be7206ebdbeed02c5f74d7ac6ce1380bf3302479c003aa91ac51a6ace2cec541`,
`4657ea81c6f0044a5c6551ce4d03b870ce2dcf832e04d225e4aa07f3d3eea160`,
`a58b5fd2ddf43e8d9af6458418148ebcc683d11d8f52b8233a54c72e46179585`.
The 80 frame SHA-line digest is
`f28e2c0f3e63df316b354afd0d318172730024ff09b49fec100b336d1ddefa39`.
The Executor's local native collector finished 05:33:29.969 UTC, created
quality/collection JSONs with missing local quality dependencies; exact
LPIPS and real image-input visual review were not yet completed at this
receipt. Its first-collected local benchmark SHA
`c3b050e1978ca014ed284a92fcd1488ebea3f6df32c0b61b5385cc5659091393`
retained all five request values and timing scope. Later local or remote
assessment must create an explicit successor hash chain.

## Independent eight-card observer (same remote host)

Admission monotonic `1871518785969868` ns to trusted finish
`1872332975365478` ns is **814.18939561 s**. The difference from
worker `duration_s` is the 0.053845 s site admission/start interval.
The remote Phase 2 observer selected 819 bounded rows, transport stdout
SHA256 `21e3c375f341ebc072ab6aaed74f05d74798f5b1ac6f355ad946e2085c602be2`;
815 sample starts were inside this run, with 8 UUIDs complete, 0 errors,
0 >1.5 s gaps (maximum interval 1.181498383 s) and 814.18939561 s
covered by adjacent valid sample starts. Raw JSONL remains on the A100
host under `telemetry/phase2/observer/samples.jsonl`.

| GPU | sampled >100 MiB s | compute-app s | util-equivalent s | util mean/peak | peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| UUID0 `cad55364` | 680.2130 | 681.0001 | 429.3427 | 52.7325% / 100% | 23,604 |
| UUID1–7, each | 0 | 0 | 0 | 0% / 0% | 0 |

Eight-card capacity during this run is 6,513.515165 GPU-s; observed
GPU0 >100 MiB occupancy is about 10.44% and NVML utilization-equivalent
about 6.59% of that node capacity. These are 1 Hz sample-hold estimates,
not exact GPU allocation, kernel/SM time, or proof of avoidable idle.
Visible `/proc` lineage from PID 2932806 gives lower bounds
2,508.78 user-s + 9,954.29 system-s and sampled summed RSS peak
14,267,346,944 bytes. NVML app PIDs
`757934,763594,770002,777374,784713` are not directly joined to
visible `/proc` PIDs because of the node's PID namespace distinction.

## R03 LPIPS CPU overlap and comparability limit

The separate authorized R03 original LPIPS judge ran
05:30:14.679948–05:30:31.856055 UTC, same-host monotonic
`1872164903584728`–`1872182079726278` ns. During this exact 17.176 s
overlap, observer lower bounds were 16.25 user+7.66 system CPU-s for the
R04 worker ancestry and 310.59 user+16.40 system CPU-s for the R03
quality ancestry; GPU0 had only about 1.129 s >100 MiB and 0.079
util-equivalent s. The remote shell saw 256 logical CPUs in affinity,
but `/sys/fs/cgroup/cpu.max` was `18000000 100000`, a 180-core quota
for the inspected cgroup; exact per-process quota inheritance was not
instrumented.

R04 p04 denoise ended 05:30:02, decode 05:30:07, pixel export 05:30:08;
p05's first model-availability log is 05:30:44, transformer loaded
05:31:05, and p05 request denoising began 05:31:34. Thus the LPIPS
CPU burst overlapped p05 **cold startup/import before its request**, not
an observed p05 measured denoise interval. It may still perturb CPU,
page cache, I/O, or model-load wall, so do not call the overall rollout
wall fully equal-condition. The five-request denominator itself has no
direct temporal LPIPS overlap; p01–p04 were already around 76.4 s versus
R03 around 74 s, so the 11.75 s R04–R03 request difference should not be
causally attributed to this short judge overlap. No repeat benchmark or
unqualified speed ranking is inferred here.
