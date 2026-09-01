# SANA-Video 2B H100 FP8 Executor

Status: **implementation VALIDATED; direct integrated candidate
ACCEPTED_BOUNDED; autonomous isolated executor delivery NOT_RUN**.

## 结论

我们已经把 FP8 做成了 Sol-Agent 可独立调度的量化 executor 组件，并接入现有
`Kernel R20 + EasyCache R12` 基线。它不是把 `--quantization fp8` 字符串塞进
启动命令，而是在模型完成 BF16 加载后，将 20 个 Transformer block 中的
`conv_inverted` 与 `conv_point` 两个 1x1 FFN 投影替换为真实的 H100 W8A8
E4M3 GEMM，共 40 个模块；其余 attention、depthwise/temporal convolution、
VAE 和 text encoder 保持原精度。

直接集成实验保留的候选为：Kernel R20 + FP8 FFN + EasyCache threshold 0.13。
相同 H100、prompt、seed 和 workload 下，它相对新鲜 BF16/cache-0.10
integrated control 的：

- warmup 后稳态请求：23.32 s -> 两次 20.99/20.98 s，中位数 20.985 s，
  **1.111x**；
- denoise：19.8883 s -> 两次 17.6059/17.5913 s，中位数 17.5986 s，
  **1.130x**；
- runtime peak：18,102 MiB -> 17,476 MiB，降低 **3.46%**；
- 被选中 FFN 权重与 scale：1,806,336,000 B -> 904,422,400 B，降低
  **49.93%**。

和上一阶段已接受的 29.2 s integrated Prompt-1 结果相比，联合候选两次 outer
envelope 中位数为 27.35 s，即 **1.068x**；相对 61.7 s dense baseline 为
**2.256x**。这两个 outer 数字包含运行时一步 warmup，且 2.256x 只覆盖当前
一个 prompt，因此不应当解释为论文级、多 prompt 的最终 claim。更可靠的
当前联合候选测量是上面的 1.111x steady-state 与 1.130x denoise。

这里不能把 1.111x/1.130x 归因为“纯 FP8 增量”：候选同时把 EasyCache
threshold 从 0.10 调到了 0.13，而 FP8-only full generation 与
BF16/cache-0.13 counterfactual 都没有运行。因此，本报告接受的是
**FP8 + cache-retune 的联合集成点**；独立 FP8 executor frontier 仍为
`NOT_RUN`。

## 为什么是“参照 NVFP4”，而不是照抄 NVFP4

官方历史源码中，NVFP4 是旧式 Symposium 的独立搜索维度，并复用通用
session/executor；新版 `orchestration/` 里没有一个可直接复制的 NVFP4
executor 目录。其核心可复用思想是“由独立量化 agent 搜索 load-time 精度
变换，并以质量门控交付”。

本实现保留了这个结构：

- `orchestration/techniques.toml` 注册 `fp8 -> quant_qe`，标记为
  `quality_gated`；
- `workflow/quant_qe/` 定义 FP8 的 scope、retention 和 delivery contract；
- `fp8_ffn` transform 独占 `FFN_PRECISION` seam；
- SANA-specific runtime 在 post-load 阶段安装量化模块；
- executor 必须提交真实 E4M3 权重、模块激活、性能和视频质量回执，静默
  BF16 fallback 不计作 FP8 结果。

没有直接复用 NVFP4 kernel，是因为 H100/SM90 不支持 Blackwell 的 native
NVFP4 路径，而且 SANA-Video 的 FFN 是 Conv2d 结构，不是 LTX 的 Linear
GELU FFN。两个 1x1 Conv2d 在数学上可以等价改写为 NHWC 最后一维 GEMM，
因此这里复用 SGLang 已有的 dynamic activation/per-output-row weight E4M3
量化和 `apply_fp8_linear` dispatcher。

## 六轮直接工程轨迹

这六轮是本次实现与集成的机器可读工程轨迹，不是一次由 Sol-Agent 自主启动、
严格遵循 isolated-FP8 scope 后产出的官方 executor trajectory。

| Round | Candidate | 关键结果 | 决策 |
| ---: | --- | --- | --- |
| 1 | 128-token component smoke | cosine 约 0.99932、rel-RMSE 约 0.0369；小 M 下仅 0.79x/0.65x | correctness 通过，继续真实 shape |
| 2 | 32,760-token component smoke | `conv_inverted` 2.321x，`conv_point` 1.216x；真实 E4M3/CUTLASS-capable H100 回执 | 进入端到端 |
| 3 | Fresh integrated BF16 OFF | 23.32 s steady，18 hits/32 computes；视频 SHA 与上一阶段 accepted run 完全相同 | 匹配控制组成立 |
| 4 | FP8 + inherited cache 0.10 | 25.36 s steady，cache 退化为 9/41 | reject：FP8 改变 cache signal |
| 5 | FP8 + cache 0.13 | 20.99 s steady，17/33，40/40 FP8 modules active | provisional integrated retain |
| 6 | 相同候选确认 | 20.98 s steady，17/33，输出 SHA 与 Round 5 相同 | integrated accepted bounded |

完整机器可读轨迹见
[`fp8_executor/TRAJECTORY.jsonl`](fp8_executor/TRAJECTORY.jsonl)，候选账本见仓库
根目录的 `candidates.jsonl` 和 `benchmark.csv`。

## H100 数值与执行证据

真实 SANA token 数 `M=21*30*52=32760` 的 component smoke：

| Projection | Shape | BF16 | FP8 | Micro speedup | Cosine | Relative RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `conv_inverted` | 32760x2240 -> 13440 | 3.7792 ms | 1.6280 ms | 2.321x | 0.999321 | 0.036843 |
| `conv_point` | 32760x6720 -> 2240 | 1.4541 ms | 1.1957 ms | 1.216x | 0.999318 | 0.036922 |

端到端日志与 benchmark 同时证明：

- GPU 为 H100 capability 9.0；
- weight dtype 为 `torch.float8_e4m3fn`；
- `cutlass_fp8_supported=true`；
- 40 个转换模块全部产生 `SANA_FP8_MODULE_ACTIVE`；
- 实际输入 shape 为 `[21,2240,30,52]` 或 `[21,6720,30,52]`；
- 没有 skipped module、fallback module 或 residual compute app。

## 计时口径

旧 wrapper 把 `GENERATE_OK` 错标为排除了 warmup；实际它包含一次 runtime
warmup。新回执已经同时写出：

- `generation_s`：首次 `gen.generate` 外层 envelope，包含一步 warmup；
- `warmup_request_s`：一步 warmup 自身；
- `warm_steady_state_s`：runtime 报告的 warmup 后请求；
- `denoise_s` 与 `decode_s`：阶段计时。

| Run | Outer | Warmup | Steady | Denoise | Decode | Cache | FP8 active | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Accepted integrated (historical) | 29.2 | 5.03 | 23.22 | 19.7843 | 3.0582 | 18/32 | 0 | accepted baseline |
| Fresh integrated OFF | 38.2 | 13.91 | 23.32 | 19.8883 | 3.0557 | 18/32 | 0 | matched control |
| FP8, cache 0.10 | 31.9 | 5.56 | 25.36 | 21.9633 | 3.0177 | 9/41 | 40 | rejected |
| FP8, cache 0.13 | 27.3 | 5.31 | 20.99 | 17.6059 | 3.0141 | 17/33 | 40 | retained |
| FP8, cache 0.13 confirm | 27.4 | 5.44 | 20.98 | 17.5913 | 3.0126 | 17/33 | 40 | accepted bounded |

Fresh OFF 的 outer 38.2 s 异常来自 13.91 s warmup；其 steady、denoise、
memory、cache schedule 和输出都复现了历史 accepted integrated run。因此选择
依据采用 matched steady/denoise，且仍只将 1.111x/1.130x 视作联合候选收益；
不把 38.2/27.35=1.397x 冒充纯 FP8 收益。

## 质量验收边界

FP8 是有损变换，本报告不声称 bitwise 或感知无损。

- component gate：所有输出 finite，minimum cosine 0.999318，maximum
  relative RMSE 0.036923，分别通过 0.995 与 0.10 阈值；
- video contract：两次 retained run 都是 832x480、81 frames、16 fps、
  5.0625 s；
- determinism：两次 retained run 的 MP4 SHA-256 都是
  `171700f5e7e4800f1ce351920dac267864756886b60bb59cb138cbdec547575f`；
- sampled visual gate：人工查看 first/middle/last，主体、森林场景、姿态和光照
  连贯，没有空帧、结构崩坏或明显量化伪影；FP8 与 BF16 有构图/光照数值漂移，
  不是像素一致。

| | First | Middle | Last |
| --- | --- | --- | --- |
| BF16 integrated OFF | ![](fp8_executor/visual/bf16_off/frame_01.png) | ![](fp8_executor/visual/bf16_off/frame_02.png) | ![](fp8_executor/visual/bf16_off/frame_03.png) |
| FP8 + cache 0.13 | ![](fp8_executor/visual/fp8_cache013/frame_01.png) | ![](fp8_executor/visual/fp8_cache013/frame_02.png) | ![](fp8_executor/visual/fp8_cache013/frame_03.png) |

VBench、LPIPS、全帧 perceptual metric、blind reviewer、第二个 prompt、
BF16/cache-0.13 counterfactual、FP8-only 端到端 ablation，以及由 autonomous
FP8 executor 产生的 schema-v2 `DELIVERY.json` 均为 **NOT_RUN**。所以结论限定为
“组件真实执行且一个固定 prompt 的联合候选通过轻量质量门”，不扩展为论文
全表、普适质量等价或纯 FP8 加速归因。

## 代码入口

- Runtime：`external/sol_runtime/python/sglang/multimodal_gen/runtime/models/dits/sana_fp8.py`
- Model hook：`external/sol_runtime/python/sglang/multimodal_gen/runtime/models/dits/sana_video.py`
- Load transform：`external/sol_runtime/python/sglang/multimodal_gen/runtime/efficiency/transforms/fp8_ffn.py`
- CLI wiring：`external/sol_runtime/scripts/sana/sana_video_sglang_run.py`
- Harness receipts：`models/sana_video_2b_h100/baseline/gpu_infer.py`
- Retained manifest：`config/sana_video_2b_h100/fp8/fp8_ffn_all_blocks_cache013.toml`
- Component smoke：`scripts/sana/fp8_component_smoke.py`
- Executor contract：`workflow/quant_qe/nodes/codex_executor/`

## 验证与证据

- Local integration unittest：9/9 passed；
- Frozen remote Python combined unittest：13/13 passed（FP8 runtime 4 +
  integration/orchestration 9）；
- Python compileall、TOML parse、FP8 orchestrator dry-run：passed；
- H100 component smoke：passed；
- matching OFF、FP8 0.10、FP8 0.13、FP8 0.13 confirmation：均完成且
  wrapper status `VALIDATED`；
- GPU 6 在结束时为 0 MiB、0 compute app，实验租约于
  `2026-09-01T12:03:27Z` 标记为 `released`；
- large videos 保留在远端 persistent project：
  `/home/jiangzhikun/yongyan_liu/Experiments/SolAgent/20260901-sana-video-fp8-executor/`；
- 四次紧凑原始 `run.log`、机器可读 timing receipts、benchmark、quality、
  collection、run-config 和视觉样本已经纳入
  `reports/fp8_executor/evidence/` 与 `reports/fp8_executor/visual/`。

机器可读最终状态见
[`FP8-SEARCH-STATE.json`](fp8_executor/FP8-SEARCH-STATE.json) 和
[`FP8-DELIVERY.json`](fp8_executor/FP8-DELIVERY.json)。后者明确是直接集成验收包，
不是官方 executor schema-v2 `DELIVERY.json`。
