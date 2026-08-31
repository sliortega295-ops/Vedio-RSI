# Executable reproduction plan

1. **Freeze preflight evidence.** Reconfirm the persistent mount, exact source/model/environment receipts, branch status, current CLI surface, and GPU ownership. Preserve hashes of both interrupted candidates.
2. **Validate compatibility layers.** Run the existing transport tests, add only missing lifecycle/error coverage, and commit the current-Codex bridge separately. Audit the CUDA 12.8 adapter, prove default imports are unchanged and SANA-only imports/JIT pass, commit it separately, and export its patch.
3. **Add the SANA 2B deployment contract.** Introduce an experiment-local model profile, dense config, fixed model-card prompts, UUID-locking local launcher, runtime/output receipts, and focused CPU/dry-run tests. Record PISA as `NOT_APPLICABLE` from the model and archived backend contracts.
4. **Freeze the dense baseline.** Revalidate GPU ownership, select exactly one idle H100 UUID, write the lease receipt, run the dense config once, validate video/frames/workload/timing/memory, and write immutable `BASELINE.json` before any executor worktree exists.
5. **Dispatch exactly two executors.** Materialize independent Kernel and Cache worktrees from the same baseline with the archived prompt/contract. Spawn one child agent per worktree. Their GPU runs block on the same UUID lock and retain every round through official convergence or hard cap.
6. **Verify and integrate.** Inspect actual worktree diffs and trajectory ledgers, recompute metrics from the frozen baseline, reject unsupported/fabricated/mismatched points, combine compatible winners, and run one integrated smoke plus a small multi-prompt sanity check.
7. **Close the evidence packet.** Commit small code/manifests/reports only, exclude weights/videos/large logs, and report every gate as VALIDATED, PARTIAL, or NOT_RUN with paths, commits, commands, metrics, stop reasons, and limitations.

The root conversation remains read-only controller/acceptor. This primary reproduction agent owns all writes, GPU launches, integration, verification, and commits. The only later child agents are Kernel Executor and Cache Executor.
