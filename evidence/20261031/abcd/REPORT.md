> 2026-10-31 发布副本：保留原报告的实验日期、结论和历史状态；私有路径已替换，未发布产物不伪装为可访问链接。文末“未推送”等措辞指报告生成时；本次发布说明见根目录 PUBLICATION-NOTE.md。

# A/B/C/D execution status

Updated 2026-10-22 07:38 UTC. Coordinator-only evidence; not an amendment to the
accepted plan in `20261021-abcd-native-plan`.

## Final experiment state

**v6 A/B/C/D are all COMPLETE.** All twelve native role processes ended with
return code zero. Each arm generated 55 regular full-workload videos, used
4 Kernel / 6 Cache / 2 integration registrations, and passed objective,
component, method/visual and actual-image-attestation checks. The local final
cross-check passed with no issues (`reports/FINAL-ACCEPTANCE.json`).

| Arm | Treatment | Complete workflow wall | Observed eight-GPU mean busy |
| --- | --- | --- | --- |
| A | Historical control flow plus common site port | 4h17m46s | 4.84% |
| B | A plus parallelism guidance | 4h05m19s | 4.55% |
| C | B plus batch FIFO executor | 3h39m29s | 5.99% |
| D | C plus scoped reuse capability | 3h46m58s | 5.47% |

All four were reaggregated with the same current analysis implementation.
Per-stage, per-card, per-candidate, CPU process, API and reference-feature
records are in `campaigns/abcd-native-20261021-v6/analysis`; twelve analysis
regression tests pass. The formal quartet span is 8h04m46s in two waves.
These are single adaptive trajectories, not a causal scheduler ablation.

All 220 candidate/baseline videos were rehashed on the remote nodes and match
their persistent copies. The fresh NVML snapshot found no GPU processes on
either node, and all four run-service daemons have exited. Additional cleanup
inspection is **REVIEW**, not an unqualified clean-process claim: A/B/C/D have
35/31/43/43 remaining UID-matched process-table entries, all in zombie state.
Sampled entries on both nodes are adopted by PID 1. They are already exited,
not active compute/GPU work; no signal was sent to the container init or
unrelated processes. See `reports/REMOTE-RETENTION.json` for complete evidence.

D exercised 263 exact reference-feature hits with 45 misses, but no eligible
cross-command persistent model reuse, reset-hook execution or explicit compiled
artifact restoration. Independent evaluation feature hits are zero. Do not
claim formal persistent-model speedup from this trajectory.

The full long-form interpretation/handoff still needs consolidation; experiment
execution, independent acceptance and all-four numerical aggregation are done.
The following notes are retained historical checkpoints, not the current state.

## Historical execution checkpoint — 2026-10-22 05:08 UTC

**v6 A/B is COMPLETE; D/C is RUNNING.** The campaign started 2026-10-21 21:51:02 UTC,
PID 587783. D/C started fresh at approximately 2026-10-22 02:08:51 UTC after
the whole first-wave barrier passed. The common cumulative-image transport repair passed
the real 101,515,154-byte/97-image upstream fixture and independent bounded
review. All four Agent input bundle digests are byte-identical to v5; worker,
scheduler, evaluator, native verifier and controller proof dependency hashes are
also unchanged. Only the common API/relay envelope and numeric request-size
audit changed, committed at b0e5c01. The release check confirmed two empty
eight-card nodes. Every new arm has fresh sessions, dense videos and caches;
no v2/v5 candidate, result, quality opinion or history is supplied. New model
downloads remain zero. See `campaigns/abcd-native-20261021-v6/RELEASE-CHECK.json`.

v6 first-wave results: each arm collected dense, six Cache, two video-generating
Kernel and two actual full-five integration bundles: 55 measured regular videos.
Their
two other Kernel registrations were diagnostics/preflight attempts and remain
counted in the four-attempt cap. Both Kernel roles delivered complete packages
and exited normally: A `k04_linear16_ffn`, B `fused_linear_ffn`. Both Cache roles
also delivered and exited normally: A retains two useful operating points and
explicitly discloses that it did not establish a third useful tier; B retains
three points and still discloses the bounded search/blind-review limits. Both
Masters completed independent source/quality/method checks, actual visual inputs,
real composition and original-entry verification. Outer final objective,
component-objective, audit-field and actual-image binding checks passed; remote
services closed and all roles exited. A terminal: COMPLETE, 15,465.839 seconds
(4h17m46s), eight-card observed NVML mean 4.8413%, 2.04922 GPU lease-hours.
B terminal: COMPLETE, 14,719.043 seconds (4h05m19s), NVML mean 4.5473%,
1.83180 GPU lease-hours. NVML coverage is over 99.97% in both arm windows.
Their independently verified five-prompt native inference speedups are A
1.4413x / 1.7312x and B 1.7413x / 2.2714x; these are not rollout wall speedups,
and the adaptive recipes and quality-change levels differ.

C/D finished fresh dense and all six Cache/two video-generating Kernel candidates,
45 measured regular videos each before integration. Both Kernel and both Cache
packages are complete and their Executors exited normally. C has also collected
both actual five-prompt integrations (55 regular videos total), passed its original
integration verifier and ended Master turn one. Outer attestation required the
same-session Master to repair structured audit fields and ambiguous multi-image
tool-call bindings; turn two is active, with no new candidate/video budget.
D has completed both `sana_integrated_fast_01` and
`sana_integrated_conservative_02` on canonical GPU0: all four arms have now
generated 55 measured regular videos each (220 total), without extra search
registrations. D's original independent integration verification has returned
success and its Master is finalizing its package. C's same-session Master is
still completing the stricter image-binding/audit repair. Both immutable composition
trees were independently inspected by the outer acceptor: only the DiT and
current-run helper modules differ, not the runner/scheduler/instrumentation.
Neither C nor D is terminally accepted yet. C retains `k04_ffn_attn16` and
`easy_seed/tea_matched/tea_seed`; D retains `k03_attn_ffn` and
`easy_matched/easy_seed/tea_seed`. D explicitly rejected its Taylor candidates
for new sampled artifacts and reports incomplete three-family matched timing;
it does not fabricate a quality winner or extra trial. No first-wave source,
opinion, timing or session is supplied. The four fresh dense sets have identical
55 sampled PNG hashes and all have five 50-step/100-logical-call traces
(`reports/BASELINE-CONSISTENCY.json`). All twelve role session IDs are distinct
across arms (`reports/SESSION-SEPARATION.json`). In-run A multimodal requests
above 64 MiB have completed successfully (including approximately 131 MiB),
confirming that the repaired envelope works beyond the preparation fixture.
Multi-minute API latency/retries and all no-ready GPU idle time remain recorded.
Network request envelopes are not model weight downloads or an exact VPN bill.

The 02:40 live diagnostic already recorded D reference-feature cache hits
(112 hits/44 misses across four evaluations), but zero model-persistence
eligibility/reuse and zero explicit compile-artifact restores at that time.
These are partial feature-coverage observations, not D's final speedup. Final
analysis preserves all per-evaluation cache events even if independent checks
overwrite the latest quality.json. True compiler cache hits are not instrumented.
Eleven analysis arithmetic/schema tests pass. The frozen experimental runtime
remains unchanged at b0e5c01 throughout all four arms.

The user authorized completing all four arms. The v2 A/B attempt, started
2026-10-21 19:22:03 UTC, is now **PARTIAL_INFRA**, not a usable comparison.
Both roles independently submitted parallel work, but the common site's NVML
ordinal mapping rejected nonzero GPU workers. Both runs were stopped with exact
PID/start/UID ownership checks; remaining owned processes and GPU processes
were zero. C/D were **NOT_RUN**. All partial source, logs and remote videos are
retained. `campaigns/abcd-native-20261021-v2/analysis` reports the failed attempt's
costs only, not a scheduling speedup. v3 is preparation-only and never started
formal model sessions. This preserved interrupted attempt is not a current run.

**v5 A/B is now PARTIAL_INFRA**, not an arm comparison. It started 2026-10-21
20:25:28 UTC and was interrupted after cumulative real-image inputs exceeded
the common local relay's 64 MiB request envelope. Both Cache sessions failed
with HTTP 413; same-session resume retained the image history and failed again.
No integrated delivery or final independent acceptance was produced. C/D never
started. Exact owned-process stop confirmed no remaining local/remote owned
processes or GPU processes. Twenty A videos and thirty B videos were retained
remotely; all raw and derived evidence remains separate from any new quartet.
v3/v4 and the next repair preflights are preparation-only.

`campaigns/abcd-native-20261021-v5/FREEZE.json` binds its four planned inputs to
HEAD `24ff96eb459439375c467cb4ecb3f8724a7260f3`, code hash
`2b40309bbd7de4517fae08dee6cdb90838aaa0044615c0801a9e692128e88884`.
The final common suite passed 107 tests in 32.223s; all eight real GPU imports,
UUID checks and arithmetic plus one compiler test passed in
`abcd-probe-import-preflight-v1`. Both nodes were verified with eight idle cards
and no application processes immediately before release. Every arm starts with
fresh dense videos, sessions, role homes, source workspace and caches.

v5 progress, not final results: A/B fresh dense sums are 634.525754s and
637.434493s (the frozen native five-request metric, not rollout wall time).
Dense evidence became locally available at 20:40:16/20:40:26 UTC; Masters then
started in fresh native sessions and independently launched their two Executors
at approximately 21:06 UTC. All six first role launches remained alive; no -9
startup recurrence was observed. Both Masters later resumed their own exited
Kernel sessions through the native path. Initial Cache family coverage has
completed full-five collection in both arms. A independently used six manual
shards with 3+2 prompts per method; B used single-prompt shards and at one point
filled eight worker slots. These are observations, not a claim that either arm
has completed or that B has a causal speedup. B has also collected one full-five
Kernel candidate and is generating its final budgeted Kernel trial. No native
integration/independent final acceptance has completed yet.

The initial Kernel probe in each arm redundantly called the non-idempotent
minimal-package initializer already installed by the common probe entry. The
Executors saw the error and independently registered revised source; B's later
full-shape synthetic profiler and benchmark probes succeeded. Such consumed
attempts remain in the ledger, not silently refunded. No parent recipe or
candidate patch was supplied. Multi-minute model API latency and one transport
retry are logged independently from GPU generation; no-ready waiting is not
attributed to FIFO scheduling.

Eight actual isolated devices now pass pinned UUID/NVML/tensor checks; GPU7
also completed a real full-50-step video with one model load and one warmup
(`abcd-device-mapping-preflight-v1`, `abcd-nonzero-video-preflight-v1`). A lifecycle
thread now owns each native role through exit, fixing the transient HTTP
handler parent-death behavior. A separately found delayed-launch ownership
race is being regression-tested before the next common freeze. Probe imports
now use the same generic, no-weight ABI/optional-package adapters as videos.

The real-video independent verifier passed before launch. No prior
optimization result, recipe, session, preparation timing or reviewer finding is
available to an experimental Master/Executor.

## Verified preparation

- Both assigned nodes have 8 A100-SXM4-40GB cards. The most recent node0 check
  found all eight at 0% utilization and 0 MiB application use; launch rechecks
  the complete UUID inventory and detects foreign occupancy.
- Both nodes reuse 20 existing pinned model files, 14,002,562,288 bytes total,
  rehashed against the immutable manifest. New model/weight download bytes and
  model transfers through the local machine: **0**. Raw videos remain remote.
- Clean runtime tree SHA-256:
  `2f986617ae0e8624672a22a45c44e48ca70d2821878119cee7f87d4f2c4bffed`.
  This is a mechanically cleaned compatible runtime, not a pristine older
  runtime claim. The official Harness snapshot remains separately auditable.
- Actual local namespaces and remote Landlock/UID/network tests deny history,
  sibling-role writes, source tampering, host secrets and external networking.
  Device ioctl and host pathname-UNIX-socket limitations are explicit receipts,
  not claimed to be eliminated by Landlock.
- The real native model transport passed tool, image and broker operations.
  Requested gpt-6.1-sol/max/priority; observed actual model gpt-6.1-sol, actual
  tier default. All arms have identical settings; actual API metadata is logged.
  Native search, host memories, plugins/apps, extra agents and MCP are disabled.
- Actual eight-device CUDA arithmetic and one real compiler probe passed.
- Uninterrupted fresh full-five preparation passed (`abcd-transport-preflight-v8`):
  one model load, one native warmup, every measured clip 50 steps and 100 logical
  model calls, clean native join, durable collection and closed service.
  Receipt wall time is 900.35s including all preparation transport/collection;
  the five native request metrics sum to 634.449s. Neither is an arm result.
- Actual reuse preparation passed on the other node (`abcd-reuse-preflight-v8`):
  two compatible requests used PID 3149060, a changed key rebuilt as PID 3155016;
  reuse flags false/true/false; all three p0 videos had identical byte hashes and
  50 steps/100 logical calls. This proves the clean fresh-request case, not an
  unmeasured stateful Cache algorithm. Reset-hook ordering has CPU regression
  coverage at the actual NativeBackend.generate boundary.
- Original LPIPS math on nonidentical synthetic images was identical for plain,
  cache-miss, exact reference-feature hit, and fresh independent evaluations.
  Independent verification never reuses an Executor score or verdict.
- Actual five-video extraction uses source indices 0,8,...80, the common fixed
  2fps grid including its endpoint. Each candidate has 55 sampled frames; the
  original stratified/worst selector chooses up to 48 pairs, deduplicated.
- The v2 common suite passed **103 tests** in 30.258s before its launch. Later
  UUID/lifecycle additions passed 105 tests; current repair suite is in progress.
  Four separate analysis arithmetic/schema tests also pass. Analysis code is outside
  experimental input and does not mutate the frozen runtime.

## Review and repairs

Completed report-only code review receipt:
`artifacts/code-review/20261021-abcd-runtime-preflight/review.json`.
Ten local lenses plus one independent validator confirmed twelve findings in
the frozen pre-fix snapshot. Its verdict was Not ready. Caller-owned fixes are
tracked separately; a completed review is not itself a passing implementation.
The external peer returned an authentication-shaped execution-context error,
not a review and not proof of account login state. Local adversarial fallback
and one complete twelve-finding validator batch were collected; none skipped.

Repairs cover image-to-output binding, transactional collection, role-specific
writes, full process-session drain, fresh independent LPIPS receipts, bounded
probe archival, preparation-only launcher modes, all-worker cleanup, runnable
FIFO dispatch, terminal quarantine failures, lifecycle tests and post-warmup
reset. The real-video LPIPS preparation additionally exposed a cache-parent
ownership defect; no formal arm had begun, so all four will use its common fix.

## Remaining

The equally repaired fresh v6 quartet is frozen; A/B is accepted. Run D/C to honest terminals;
independently accept; generate fine-grained stage,
CPU and per-card reports. Failed preparation attempts are preserved separately.
Memory,
latest-method search, cross-model transfer and paper-level quality remain NOT_RUN.
