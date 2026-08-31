# Cache executor summary

The Cache search genuinely converged at round 12 of the official 20-round cap. It produced faithful TeaCache, EasyCache, and TaylorSeer implementations, measured all three families at three matched end-to-end latency tiers, preserved an invalid first attempt and all dominated/calibration points, and selected a distinct EasyCache configuration at each tier.

| Tier | Selected run | Total | Speedup | Cache work | Fixed-prompt screen |
| --- | --- | ---: | ---: | ---: | --- |
| Conservative | EasyCache round 2, threshold 0.05 | 46.2 s | 1.3355x | 9 hits / 41 computes | Quality leader or bounded tie; low drift |
| Balanced | EasyCache round 12, threshold 0.10 | 38.8 s | 1.5902x | 17 / 33 | Clear matched-band winner; very low drift |
| Aggressive | EasyCache round 9, threshold 0.25 | 28.4 s | 2.1725x | 29 / 21 | Valid with pose/framing drift; tentative edge or tie with TeaCache |

All selected outputs are real H100 runs with job-started provenance, 832x480/81-frame/16-fps videos, benchmark JSON, assessment and visual receipts, and preserved video hashes. Peak runtime memory is 17,770 MiB for each selected EasyCache point. The timing denominator remains the immutable 61.7 s dense baseline; its legacy timing-scope label carries the already disclosed one-step-warmup variance caveat.

The delivery is valid for the approved relaxed reproduction protocol: one frozen prompt plus lightweight first/middle/last validity and visual screening. The archived workflow's stricter five-prompt and independent blind-review gate is explicitly `NOT_RUN`; the master-owned small multi-prompt sanity check remains required before final integration claims.

Handoff limitation: the child committed the full search and frontier but timed out while writing the delivery documents. The primary integrator independently audited the committed ledger and artifacts, constructed this bundle without changing any search result, and retired the child. Search success and handoff failure are reported separately.
