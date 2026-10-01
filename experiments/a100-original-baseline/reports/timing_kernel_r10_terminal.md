# Kernel R10 terminal stage account

R10 `runs/20260913-143436-kernel_compile_bf16_max_autotune_ffn_all_linear` was one normal, non-overlapping UUID0 candidate. The host broker accepted it before the 14:48:39.836 UTC submit cutoff after checking all ten previous Kernel starts terminal. Remote site-job worker PID2846112 began 14:36:00.060 UTC and ended 14:58:16.802, exit0, `timed_out_budget=false`, worker wall **1336.672 s**. Checked original remote archive SHA256 `89a0009a25c4e4c045ed25ac285c29f8f832eb32b30fd8d702c1f770116cd307`; raw `benchmark.json` SHA256 `3075b9bc68f4728276fc3473ac1b60798aebc42e5c67e013fe264081c3d22f15`. Five 832×480, 81-frame, 16-fps videos and the frozen timing scope were checked.

| Prompt | Observed process UTC | Allocated GPU-s | 1 Hz util-equivalent GPU-s | Startup→import s | Build incl. load s | Opaque `generate()` incl. warmup/compile/save s | Post-generate→exit s | Source request / denoise / decode s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| p01 | 14:36:01.104–14:41:26.847 | 325.743 | 89.50 | 12.575 | 54.110 | 245.799 | 12.913 | 72.67 / 65.751 / 6.218 |
| p02 | 14:41:27.380–14:45:42.573 | 255.193 | 87.21 | 12.996 | 54.803 | 173.720 | 13.489 | 72.33 / 65.436 / 6.189 |
| p03 | 14:45:43.105–14:49:53.227 | 250.122 | 89.54 | 12.549 | 52.290 | 172.716 | 12.314 | 72.32 / 65.417 / 6.198 |
| p04 | 14:49:53.586–14:54:02.020 | 248.434 | 91.63 | 12.709 | 52.502 | 170.467 | 12.471 | 72.38 / 65.478 / 6.202 |
| p05 | 14:54:02.431–14:58:14.449 | 252.018 | 91.85 | 13.674 | 55.428 | 169.942 | 12.778 | 72.33 / 65.433 / 6.196 |
| **Sum** | five sequential UUID0 processes | **1331.510** | **449.73** | **64.502** | **269.133** | **932.644** | **63.965** | **362.03 / 327.515 / 31.002** |

Small registry-marker intervals (~1.27 s summed), worker orchestration, and transfer are not assigned to the four compact stage columns. The source-reported request/denoise/decode fields are nested and **not additive independent wall stages**. The wrapper has no separate export boundary; do not infer one by subtraction from `generate()`. The first `generate()` includes an opaque cold path. Five visible process CPU lower bounds sum to at least **2955.77 user + 9040.05 system CPU-s**; NVML app PIDs are not reliably joinable to the visible `/proc`, so full CPU attribution is unknown. Sampled utilization-weighted seconds are not exact SM occupancy.

R10 used the same submitted source SHA256 `309b56ec78bbd92e29726274501c08798f15c7734c0da6d043dca6b64a62fc2c` as R9 and activated both FFN pointwise and temporal linear branches; a single active config was measured. Its raw request sum **362.03 s** is **3.64 s / 1.016% slower** than R9's 358.39 s on the same frozen five-prompt metric. Executor rejected R10 at14:59:50 UTC; it is neither a frontier replacement nor a DELIVERY. Raw joined per-prompt event/resource rows, including exact host/UUID/PID and CPU/I/O samples, are in `reports/timing_terminal.json`.
