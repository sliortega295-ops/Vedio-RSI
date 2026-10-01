# Eight-A100 capacity snapshot through 12:28:56.990 UTC (provisional)

From immutable first-GPU `09:09:39.836269 UTC` through the checked observer snapshot, remote same-host monotonic wall is **11,956.786 s**. Across eight assigned UUIDs this exposes **95,654.291 GPU-s**. Exact job start/end and open R4 start delimit **10,585.623 distinct allocated GPU-s** (**11.0665%** of capacity); 1-Hz NVML utilization-weighted integration estimates **3,737.86 GPU-s** (**3.9077%** of capacity). The latter is *not* exact SM-active time. No observer error or incomplete-UUID row was found in 11,956 samples; five intersample gaps exceed1.5s, maximum1.8574s. Mean observer sampling wall overhead was0.41670s/sample, not CPU occupancy.

| Concurrent allocated UUIDs | Remote monotonic wall seconds | Meaning |
|---:|---:|---|
| 0 | 2,917.461 | no assigned job allocated; includes baseline/search control windows, not automatically avoidable idle |
| 1 | 7,493.027 | one allocated slot; includes sequential dense and prolonged R4 CPU cold phase |
| 2 | 1,546.298 | two distinct slots; earlier Cache/Kernel overlaps and the short Master-authorized R4/R5 overlap |

UUID0 `GPU-cad55364-e014-c35f-337e-db899afd1c6a`: allocated **8,718.343 GPU-s**, sampled util-equivalent **2,814.62 GPU-s**, memory peak23,536MiB, mean power151.84W over full observed window. UUID1 `GPU-a904c6f6-6410-dc0f-c7c3-efc068004617`: allocated **1,867.280 GPU-s**, util-equivalent **923.24 GPU-s**, memory peak23,892MiB, mean power79.89W. UUID2–7 each: **0 allocated GPU-s, 0 sampled nonzero-util samples**, idle mean power roughly53.67–56.21W. The other six cards were *available*, not assigned to six independent candidates; archived Master/Executor scheduling, Cache's missing quality dependency, and five-sequential-prompt contract must be considered before calling their capacity avoidable waste. R4's allocated-but-low-utilized GPU0 time includes active CPU compilation and is a different category from those unallocated cards.

Source: checked archive `telemetry/post_kernel_r5_1229` SHA256 `9f3fd500cecd7b05d22afc55905d09239985d37f3823b39e17c9f698bc46a4f6`, raw sample/event files under it, and reproducible `reports/full_gpu_window_post_r5_1229.json` from `scripts/analyze_full_gpu_window.py`. R4 is open at snapshot, so this is **not final-window utilization**; rerun after terminal state. A snapshot percentage cannot reconstruct unseen subsecond activity or distinguish necessary compilation from an avoidable stall without agent/job evidence.
