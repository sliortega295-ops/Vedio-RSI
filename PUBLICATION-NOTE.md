# Publication note

`reports/FINAL-REPORT.md` says no branch had been pushed. That sentence was true
when the historical closeout report was generated and is now intentionally kept
as an immutable point-in-time statement.

The durable public branches created afterward are:

| Branch | Head before RolloutBench publication | Role |
| --- | --- | --- |
| `repro/sana-video-2b-full-exploration` | `1a35329a821a3e12631e23573622b18c090176bd` | recovered harness and full exploration |
| `repro/sana2b-kernel-executor` | `e8684f3fa9077d1387de44bbb0521a38ac6b7097` | Kernel delivery |
| `repro/sana2b-cache-executor` | `e7cf11c877a91220af2f2ea2cc5e38000c0765f8` | Cache delivery |
| `repro/sana2b-kernel-cache-integrated` | `8b066889950b638b59b5703c39909b4ac0bf9cca` | two-prompt integration closeout |
| `repro/sana2b-fp8-executor` | `0565da7445489d925e937ca0661d9a059fab0f62` | separate FP8 component, outside RolloutBench v0 |

The benchmark implementation is published on
`benchmark/sol-rolloutbench-v0`. Large videos, weights, environments, runtime
logs and H100 ledgers are not uploaded to GitHub; their absence is not evidence
that the corresponding `NOT_RUN` benchmark stages were executed.
