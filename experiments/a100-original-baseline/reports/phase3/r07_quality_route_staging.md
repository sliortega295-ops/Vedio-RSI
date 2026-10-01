# R07 exact offline LPIPS route — staged, NOT_RUN

At 2026-09-14 07:52 UTC, the original Cache Executor's real R07 run was
hash-pinned for one possible CPU-only assessment by the archived LPIPS judge.
The staged coordinator source has **not** been synchronized to the remote node
or loaded by the active site-job broker; no R07 LPIPS assessment, broker
restart, GPU job, frontier gate, or visual review is authorized by this receipt.

Run: `repo/output/experiments/sana2ba100-cache_ca-0002/worktree/runs/20260914-072228-cache_ca_r07_teacache_t00087`.
The trusted site worker admitted it at 07:24:44.458584Z and finished exit 0
at 07:40:05.857301Z. The raw remote runner benchmark SHA is
`3f582161803def77d253506d8381dc1f0a6ed5c966de254d4dd0a5247dbac691`;
remote/local `run_config.json` SHA is
`cdf42836a2de2044f42409fbe94fe9ea1990e73eeb6050d67a8dbad19a93a3e8`.
The native local post-collection benchmark has five `request_s` values
96.30/95.92/95.94/95.94/96.08, sum and `total_s` **480.18**, under the
frozen five-prompt SGLang request timing scope. Five real MP4 SHA values
(p01–p05) are `a386f4085536e9608cf1b092e8ba32ec965d269e97209fee286362652397874b`,
`919f907c358d4686a7a42ff7deaee009869a9a33987498f60e9a8847bd9a957a`,
`2f329004393b0904a1e37b7dd0b1c576dc3381e7288328724d680f8ed5b5b401`,
`d63f00778e8d8ddba03a1b8b1f59b7052b47ba2d1dbd72bf0e6492a56e2b5528`,
and `5d689167939e2b9abe2eb0c5b34ffcfd00155eadc54da1ad3108e8f2aca8a818`.
The remote and local **80** aligned sampled frames (16/prompt) have the
trusted worker's content-only ordered `sha256(frame)+newline` digest
`46faed6d9430a46aad6c7986cd882c55673d7b07e0fd81c018204e3b91cc4b54`.
This pins the input set; the original collector normally selects at most 48
pairs for LPIPS, not necessarily all 80. See
`reports/phase3/r07_execution_observer.md` for independent GPU/CPU timing.

Current exact local **preassessment** file SHA256, twice rechecked after the
native local `plan_eval` completed:

| Relative path | SHA256 |
| --- | --- |
| `launch.sh` | `3feae5a534052e7dd2ba4bc5387a41bdc207e0706081fecfdb73345339c46548` |
| `manifest.resolved.toml` | `32edc63c661b28f0dd6c63b09a47d3e388a272a4451f2cf6f54ef2943f345a09` |
| `metadata.json` | `0ab88d8a3f77ff15d39ae6c6bf56076b88a1c3b60d2a8dd788376c6c63fe1d29` |
| `outputs/benchmark.json` | `ee4d80b76c56cfa9a2eab2b5921949204411a5dc7e5ab149f9b92a09514524e4` |
| `outputs/run_config.json` | `cdf42836a2de2044f42409fbe94fe9ea1990e73eeb6050d67a8dbad19a93a3e8` |
| `outputs/quality.json` | `3125086633bd8454bcc4072b921201fd7ab47537009ec85144b483154637315c` |
| `outputs/collection.json` | `223cc39506bf46bc9194bdbc5fb1b819caf0971a7966bbc571fc575b2921230a` |
| `assess_verdict.json` | `efdcb228ccb39432428292031e307dc69bb487a8dbf5a1c90997697a6f4edf91` |

The trusted worker still pins the original collector SHA
`34d85bf04eb5ee603c6c05964d0137a426c6710e9cd828ae04b54b66c05e158f`,
original `plan_eval.py` SHA `a06d9e8f845cd269cd80f3b085c5c03675129f145801ed77e9fe1e5d48ad4840`,
original LPIPS judge SHA `f68f7c72f31ec8018176e7b9be7047eba97524b73053842d99ba69ed3e1570bc`,
network-deny guard SHA `f2d1a27030f0456380f410d4bb7e118e628d543730718f74c038ddf292e71fe2`,
interpreter SHA `7c9e72ea1f06138ed5f52f49894c62f29e51e302f0d02f8fb18490e90e994331`,
official LPIPS wheel SHA `fd537af5828b69d2e6ffc0a397bd506dbc28ca183543617690844c08e102ec5e`,
LPIPS linear weight SHA `df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0`,
and AlexNet backbone SHA `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`.
The dense reference frame digest remains
`728ddacbaa5c9656cf7952e60c835192de70f64a5db8c903f192fd148ed4c964`.
Remote execution, if separately approved, hides CUDA, denies socket connects,
uses project scratch, snapshots original bytes before/after, and syncs only
small JSON/receipt files. It must not regenerate video or alter dense.

Staged local source SHA256:

- `repo/orchestration/site_quality_worker.py`:
  `374b56dd1a47497a8c2879c5ffa4a1ddc544b4a68342c80c1cfb7a66b017693c`
- `repo/orchestration/site_quality_transport.py`:
  `0a6f65895e228fdd4e872d7a0de21df4cfe5cb556a5b32246d462724b28605f4`
- `repo/tests/test_site_quality_route.py`:
  `98b867f9e797f47921876e98ed13ae7ace3071fb14fd665b0c4fcfd2448a3dbf`

`PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_site_quality_route -v`
passed **20/20** in 0.088 s, including exact R07 local bundle/80-frame
identity and wrong-benchmark-SHA rejection before SSH. Existing negative
tests also reject wrong run ID, wrong trusted worker/dependency source SHA,
network connects, and repeated R01–R06 assessments without an explicitly
approved post-hash chain. `git diff --check` passed. These are CPU/static
tests, not a remote E2E judge result.

The native local `quality.json` has no valid LPIPS yet and retains the raw
Gemini blocker. Cache9 attempt9 had **only R02** in its actual `--image`
transport, so its R07 visual PASS claim is externally UNSUPPORTED. Numerical
LPIPS, if later obtained, cannot itself promote R07. Before any activation,
recheck all exact local/remote input and trusted-source SHA, typed queues
pending/unresolved 0, no active assessor or candidate GPU worker, back up the
prior remote worker, sync only the pinned coordinator worker, scoped restart
only the host site-job broker, and verify new PID/heartbeat/SHA. Then permit
at most one typed CPU R07 assessment with pre/post receipts. Intervening
Agent mutation fails closed and requires new review.
