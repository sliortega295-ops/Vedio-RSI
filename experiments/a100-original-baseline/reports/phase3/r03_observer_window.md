# R03 Phase 2 observer window (independent resource sample integration)

Read-only analysis at 2026-09-14 05:18:33 UTC. This receipt covers the
remote **budget admission** 04:53:50.441482 UTC to trusted worker finish
05:07:13.650173 UTC on the same A100 host. Remote monotonic window:
`1869980665151486`–`1870783873821613` ns = **803.208670127 s**.
The worker itself started about 0.063336 s after admission and its trusted
status gives 803.145334299 s. No local-host monotonic value is subtracted.

Raw source remains remote at
`/share/project/shuchenweng/yongyan_liu/Experiments/SolAgent/20260913-a100-original-baseline/telemetry/phase2/observer/samples.jsonl`.
The bounded read returned 808 JSONL rows (804 sample starts inside the
window), transport stdout SHA256
`6242f2b6d257344d4fdf27c914291067584585e5ea3d6a9ca1ce7e3406b34c5c`.
Remote raw file stat at read: 8,515,720 bytes, mtime Unix 1789363112;
the file continues to grow, so this is not a whole-file SHA. Within the
R03 window there are 0 error rows, 0 incomplete-eight-UUID rows, no
>1.5 s sample-start interval (maximum 1.328009462 s), and
803.208670127/803.208670127 s covered by adjacent valid samples.
The observer's two earlier 1.7435/1.8636 s gaps occurred before this job;
they remain in the full-window audit.

| GPU index / UUID suffix | >100 MiB sampled seconds | NVML util-equivalent seconds | sampled util mean / peak | peak memory MiB | compute-app sampled seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 / `cad55364` | 667.0618 | 414.1875 | 51.5666% / 100% | 23,892 | 670.0000 |
| 1 / `a904c6f6` | 0 | 0 | 0% / 0% | 0 | 0 |
| 2 / `79ab3e5e` | 0 | 0 | 0% / 0% | 0 | 0 |
| 3 / `bab43db0` | 0 | 0 | 0% / 0% | 0 | 0 |
| 4 / `44bddf94` | 0 | 0 | 0% / 0% | 0 | 0 |
| 5 / `af8dba1f` | 0 | 0 | 0% / 0% | 0 | 0 |
| 6 / `5edaa614` | 0 | 0 | 0% / 0% | 0 | 0 |
| 7 / `01fff32a` | 0 | 0 | 0% / 0% | 0 | 0 |

The eight-card window has 6,425.669361 GPU-s of capacity. GPU0's
>100 MiB sample-hold estimate is 10.38% of that capacity, and its
util-equivalent estimate 6.45%. These are *not* exact allocation,
kernel execution, or SM busy fractions. The trusted run occupied one
assigned GPU; the other seven had no observed app/memory/util during
this run. That is a resource-state observation, not by itself a claim of
avoidable scheduling inefficiency. The five prompt processes restarted
sequentially, leaving GPU0 without >100 MiB during about 136 s of the
admission window; setup/warmup/export attribution needs run log bounds.

NVML compute-app PIDs sampled on GPU0 were
`697078, 703983, 709820, 715050, 720457`; these are not assumed to
match visible `/proc` PIDs. Independently, observer `/proc` ancestry from
trusted site-job worker PID 2912395 yielded sampled CPU lower bounds
of **2,526.10 user-s + 9,838.70 system-s**, peak sampled summed RSS
14,165,323,776 bytes, sampled read I/O lower bound 0 bytes and write I/O
lower bound 64,069,632 bytes. The top five visible multiprocessing scheduler
PIDs were `2918568, 2915966, 2919565, 2913177, 2917566`;
their CPU dominates the lineage sum. CPU accounting is per `(PID,start_ticks)`
and lower-bound because short-lived processes can fall between samples.
It is not an NVML-PID join, and Agent/Master local CPU is accounted separately.

Analysis code:
`scripts/analyze_phase2_observer.py` SHA256
`558f7070ab4134c30ba79fb3d030caa3cc916783c3f3d30a9be74205e96233ad`;
it reads only the exact Phase 2 observer file remotely, filters around the
trusted same-host monotonic window, uses left-sample hold between adjacent
valid samples, and leaves >1.5 s intervals unobserved. The script itself
does not change remote or candidate state.
