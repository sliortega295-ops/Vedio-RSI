# FastRoll: video optimization execution harness

## 定位与状态

这是一版**可运行的 Codex Skill + 批量执行后端**，不是另一个 LLM，
也不是旧 OptRoll 调度脚本的重命名。Codex 提出候选，FastRoll 展开独立输入、
自动分配资源、维护长驻 worker、复用可验证的 reference 证据并收集结果。

本 PR 不修改 `benchmark/sol-rolloutbench-v0`、历史实验或报告 PR #1。
这些分支的运行/评测合同仍保持原状。主分支原 README 也不覆盖。

**验证状态：CPU 单元与真实子进程集成测试已执行；GPU / SANA 接入、
实际 Codex 会话、真实故障注入与加速实验均 NOT_RUN。** SANA 适配器根据
仓库 `09b963065a1ef876612fde8be2150eae68a6334d` 的 SGLang API 编写，
不是声称已在 BAAI 验收的生产部署。没有下载权重、消耗模型 API 或启动 GPU。

## 完整执行链

- 多 Agent 可向同一工作目录提交不可变 batch；同一 ID 重复提交幂等，内容冲突拒绝。
- DAG 只调度依赖已满足的任务；GPU 不按 Kernel/Cache 固定分区，不让队首的
  affinity 阻塞其他可运行任务。独立 prompt/seed 可在多卡上同时执行。
- CPU 与 GPU 各有 worker slot；一个 Agent 可以提交很多任务，不需要一任务一 Agent。
- 相同 session 优先复用 worker；aging 防止亲和性无限饿死其他就绪任务。
- one-shot primary 与 resident 质量生成严格分离，不能用热运行替换冷启动排名。
- 仅 reference generation/standalone scoring 可缓存；缓存绑定完整 session、
  模型/评估资产、输入内容、采样参数、设备及 Harness 版本。
- SQLite WAL 持久化队列；状态转移和哈希链事件原子写入。输出绑定真实文件哈希。
- 超时/取消终止本任务进程组；失败废弃 resident session，有限重试并阻塞失败后继。
- Controller 可断开前端运行；恢复时只重试未完成任务，不伪称外部副作用 exactly-once。
- 显式外部授权、期限、source 范围和物理 UUID 锁；默认不根据空闲状态擅自运行。

## Quick start: no GPU, no API, no network

Linux and Python >= 3.11; the control plane uses only the standard library.
Run from the checkout, or `pip install -e .` in a prepared environment.

```bash
python -m unittest discover -s tests -v
python -m fastroll submit --workspace /tmp/fastroll-demo --plan examples/fastroll-demo.json
python -m fastroll run --workspace /tmp/fastroll-demo --simulate-gpus 2 --idle-exit-s 1
python -m fastroll verify --workspace /tmp/fastroll-demo
python -m fastroll collect --workspace /tmp/fastroll-demo
```

`SIM-*` are CPU child processes with CUDA hidden. Simulation accepts only the
built-in demo and compare adapters; it cannot execute a supplied command/module.
Every result is labelled simulation. These durations are NOT video performance.
Use a fresh directory for a new independent demo; resubmission is idempotent.

## Agent-facing tools

```bash
python -m fastroll expand --matrix matrix.json --output plan.json
python -m fastroll submit --workspace /path/to/state --plan plan.json
python -m fastroll sessions --workspace /path/to/state
python -m fastroll start --workspace /path/to/state --authorization /operator/owner.json
python -m fastroll status --workspace /path/to/state
python -m fastroll wait --workspace /path/to/state --timeout 7200
python -m fastroll collect --workspace /path/to/state > collected.json
python -m fastroll cancel --workspace /path/to/state --batch candidate-batch
```

`start` detaches one local controller; it does not SSH elsewhere. Execute it ON
THE AUTHORIZED COMPUTE HOST in the appropriate Python environment. SPAWNED is
not READY: check `controller.log`. The controller accepts new batches while
running and exits after `--idle-exit-s` without work (default 30 seconds).
An unreachable dependency/device leaves tasks queued, never fabricated as done.
`wait` timing out does not cancel work. Explicit cancel or authority revocation does.

`status` shows counts and a bounded ready/dependency preview, avoiding megabytes
of receipts in the Agent context. `collect` returns full verified results.
The Skill is `.agents/skills/video-fast-rollout/SKILL.md`; no API credential is
required by FastRoll. Native Codex decides proposals, searches and subagents.

## Task contract

A plan has exactly `schema_version: 1`, `batch_id`, and `tasks`. Each task has:

| Field | Meaning |
|---|---|
| `id` | Safe ID, unique inside batch |
| `role` | primary / generate / reference_generate / reference_score / score / validate / build |
| `resource` | cpu or gpu; a GPU task uses ONE full physical device |
| `session` | backend, config, optional Python interpreter, pinned source and assets |
| `payload` | Sampling parameters or evaluation inputs; explicit seed for SANA |
| `depends_on`, `inputs` | Real dependencies and named predecessor artifact bindings |
| `mode` | oneshot or resident; primary must be oneshot |
| `reuse` | Only legal for reference_generate and reference_score |
| `priority`, `estimate_s` | Scheduling hints, not measured performance |
| `timeout_s`, `retries` | End-to-end task deadline and at most three retries |
| `min_free_mib`, `device` | Cold admission memory requirement; optional strict UUID affinity |

Inputs use `{"task":"predecessor-id","artifact":"video"}`. This is dataflow,
not a path typed by an LLM. The producer must have succeeded and its artifact
must still match its receipt. Different batch tasks currently cannot depend on
one another; submit a single DAG when there is a causal dependency.

The matrix helper accepts `batch_id`, `samples` (sampling payloads), `candidates`
(id/session/mode/optional depends_on), optional `reference_session`, and optional
`paired_evaluator` (session/resource/payload). It creates one independent task per
sample, shared reference nodes, and paired-evaluation dependencies. It does not
invent future candidates or remove sequential search dependencies.

## Real source / asset preparation

Source execution requires a clean Git snapshot, including no untracked files.
Put every candidate in a separate worktree; never edit a running candidate.
`pin-source` records its commit, full tracked-file inventory and content hashes.
Symlinks/submodules in source must first be resolved into separate snapshots.
Model snapshots may contain Hugging Face blob symlinks; assets pin file contents.

```bash
python -m fastroll pin-source --root /path/to/sol_runtime > source.json
python -m fastroll pin-assets --root /path/to/local/model-snapshot > model-assets.json
python examples/make_sana_plan.py --runtime /path/to/sol_runtime \
  --model /path/to/local/model-snapshot --output /tmp/sana-plan.json
```

Preparation hashes existing files; it does not download them. SANA requires its
whole `model_path` to be covered by an asset manifest. Other adapters must declare
ALL external model/evaluator weights in `session.assets`, not merely a filename.
Asset hashes are reused only while inode/size/mtime/ctime remain unchanged; this
assumes a trusted local filesystem. Imported environment dependencies must remain
fixed during a campaign; use a pinned environment/container at the site.

The example makes an isolated primary plus four independent dense samples. It
only generates/validates video shape, NOT video quality. Integrate a real evaluator
using command/Python adapters; there is no fake or silently substituted VBench gate.

## Authorization and GPU ownership

`fastroll sessions` prints an INACTIVE template. Only the resource owner may
approve it. Store approved grants outside Git. Minimum fields are:

```json
{
  "authorized": false,
  "issuer": "RESOURCE_OWNER_MUST_APPROVE",
  "workspace": "/absolute/state/path",
  "gpu_uuids": ["GPU-EXACT-UUID"],
  "source_roots": ["/absolute/candidate-worktrees"],
  "approved_sessions": ["EXACT_NORMALIZED_SESSION_HASH_FROM_SESSIONS"],
  "not_before": 0,
  "expires_at": 0,
  "lock_root": "/tmp/vedio-rsi-gpu-locks"
}
```

Times are UTC Unix seconds; expiry is continuously checked. Changing/removing the
file revokes current work. By default only exact sessions are allowed. For a
bounded autonomous search, an owner can explicitly set
`allow_new_sessions_under_source_roots: true`, list `approved_backends`, and
restrict `source_roots` to candidate worktrees. This allows new pinned candidate
sessions without a human signing each hypothesis. It is NOT enabled by the Skill.

All FastRoll controllers on a host must use the SAME operator-managed lock root.
These advisory locks do not coordinate with an unrelated Slurm allocation or
another program's differently named locks. Use a real scheduler reservation or
exclusive host allocation as the outer boundary. Do not nest the old
GPU-locking `launch_config` wrapper inside a FastRoll task; use a lock-free adapter
entrypoint. Locks are held while resident models are alive and released on exit.
No MIG, multi-node RPC, tensor-parallel gang scheduling, or fractional GPU sharing
is claimed in this release. A100/H100 UUID pools are supported by the generic
single-node dispatcher; each model backend still requires site validation.

## Adapter contract

### Command adapter: existing executors and evaluators

Session backend `command`; config `argv` is a string array, not shell text.
Known placeholders: `{request}`, `{result}`, `{output_dir}`. Run code from a pinned
source. The executable must read the input JSON and write this response:

```json
{
  "request_id": "COPY_FROM_REQUEST",
  "metrics": {"lpips": 0.05},
  "artifacts": {"scores": "/absolute/attempt-directory/scores.json"}
}
```

The number above is schema illustration, not an experiment. Nonzero exit, missing
result, wrong request ID, empty artifact, path escape or changed evidence fails.
All output artifacts must be under that attempt directory. Put scalar metrics
into an artifact too when a downstream compare task consumes them. A successful
process is not an independent quality verdict. `compare` applies explicit numeric
thresholds to real upstream score artifacts; it does not accept textual claims.

### Python adapter: resident evaluators or custom generators

Session backend `python`; config `module` is a source-relative file, `factory`
defaults to `Adapter`, and `options` supplies its constructor parameters. It must
implement `reset(request)`, `run(request)`, `close()`. Its result format is the same
as above. Each resident process has exactly one immutable session. Source/config/
asset changes cause replacement, NOT arbitrary in-process code hot swapping.

### SANA adapter

Session backend `sana` uses the pinned Sol SGLang `DiffGenerator` API. It loads a
local snapshot, applies only explicit recipe environment values, isolates ports
per physical index, passes seed per request, and checks output dimensions/frame
count with ffprobe. Recipe flags are fixed for the lifetime of a worker. SANA
artifacts have `quality_validated: false` until the separate evaluator runs.

Do not treat FP32-to-BF16 as bitwise lossless, or decide quality from shape checks.
This implementation is source-integrated, but real SGLang/CUDA acceptance remains
NOT_RUN. There is no fallback to a different model/runtime on an import failure.

## Enabling residency safely

Residency is available now, but disabled for real adapters without a functional
certificate. Run a separate, explicitly authorized qualification workspace:

```bash
python -m fastroll qualify-reset --session session-without-certificate.json \
  --samples two-sampling-payloads.json --workspace /path/to/qualification \
  --authorization /operator/qualification-owner.json
```

For every authorized device it runs fresh A, B, A and resident A, B, A. Samples
must differ in prompt/seed and agree in other parameters. Videos are compared by
full decoded-frame hashes (ffmpeg required), not MP4 container metadata. Generic
artifacts use byte hashes. All six receipts per device are retained. Only a pass
produces `reset-certificate.json`; attach its path/hash as
`session.config.reset_certificate`, then obtain authorization for the normalized
new session hash (or an explicit autonomous-source grant). Preparation is not
permission to launch qualification. Simulation certificates cannot enable GPUs.

At admission the certificate matrix is replayed against source/recipe/session,
sampling-domain, device and output identities. A changed source/asset or a new
sampling domain needs requalification. This is a bounded functional test, not a
proof of equality for every possible future prompt.

## Reference reuse and timing

Reuse keys include the exact session, all declared assets, sampling payload,
input hashes, physical device and Harness source fingerprint. CPU slots also have
conservative separate keys. No cross-device numerical equivalence is presumed.
Duplicate in-flight keys are coalesced. Consumers point to the original producer;
we do not generate a new fake execution receipt or cache a final candidate verdict.

Primary is ALWAYS one-shot. `primary_wall_s` includes source checks, worker boot,
model initialization, execution, output validation and process shutdown. Auxiliary
quality generation may be resident. Startup/reset/execute times and cache producers
are separate provenance fields. `attempt_gpu_hours` counts wall occupancy, not SM
utilization; controller events also record full reserved GPU-seconds, including
idle residency. Timings in this harness are not silently interchangeable with
legacy Sol `total_s`, worker wall or frozen RolloutBench TTVF.

## Recovery, security and boundaries

SQLite is for local disk, not an arbitrary network filesystem. One controller
holds the workspace lock; multiple producers may submit concurrently. Terminal
commits are durable. After a crash unfinished attempts are retried in new output
directories; already completed results are verified, not rerun. External effects
are at-least-once, not exactly-once. Adapters must be idempotent and must not spawn
detached processes outside their owned process group.

A worker watches controller death; normal timeout/cancellation terminates its
entire process group. An expired/revoked grant stops admission and active tasks.
SIGKILL/node failure can still require site cleanup; no controlled GPU failure
injection was performed in this PR. Never kill another user's process to get a GPU.

Process isolation is NOT a security sandbox. Arbitrary GitHub code needs source
review or an operator-provisioned container/VM. Offline environment flags disable
normal model auto-downloads but are not network egress enforcement. API secrets
are not inherited by workers; filesystem isolation remains the site's job.

## References and evidence

- Codex Skill authoring: https://developers.openai.com/codex/skills
- Noninteractive Codex: https://developers.openai.com/codex/noninteractive
- Pinned runtime API inspected: https://github.com/sliortega295-ops/Vedio-RSI/blob/09b963065a1ef876612fde8be2150eae68a6334d/external/sol_runtime/scripts/sana/sana_video_sglang_run.py
- Prior evidence remains on PR #1 and `benchmark/sol-rolloutbench-v0`.

This PR delivers code, tests, a native Skill and operating instructions together.
It does not deliver a new speedup, a completed A100 integration, or an RSI claim.
