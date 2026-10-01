# 当前结果能支持什么论文结论

发布时间：2026-10-01。这里总结的是截至 2026-09-14 留存的实验结果；此次发布没有新增 GPU 实验。

**已经有支撑研究动机和系统诊断的初步证据，还没有支撑“我们的方法显著提高搜索效率”的正结果。** 目前最有价值的发现是：Agent 选出的候选即使生成视频更快，也可能花更长时间才能完成运行和验收。因此 rollout 需要同时衡量运行成本、依赖关系和验收可信度。

## 三组证据各自说明什么

| 实验 | 实测结果 | 可以支持的表述 | 表述边界 |
| --- | --- | --- | --- |
| 原版 orchestration-lite Agent，A100，5 prompts、seed 42 | Dense 五请求合计 578.02 s；Kernel R9 为 358.39 s，1.6128×；Cache R05/R08 为 371.36/480.96 s，1.5565×/1.2018× | 原版 Master/Executor 在适配后的 A100 环境中形成了真实候选与单项验收轨迹 | Kernel 的 lossless 是原合同允许 BF16 精度变化的结构/方法判断，未证明逐位一致；组合没有运行，整个 baseline 仍为 PARTIAL |
| A100 的冷进程运行时间 | Dense worker 991.354 s；Kernel R9 worker 1342.793 s；R3 请求 380.28 s，但 worker 2005.115 s | 仅报生成请求时间会漏掉影响候选决策效率的重要成本 | 当前适配器每个 prompt 重启进程，加载/冷启动成本包含站点适配影响；不能直接归因给未经适配的原版 Sol-Agent |
| H100 的同候选调度 pilot | serial1 TTVF 中位数 6987.23 s；FIFO2 6337.06 s；OptRoll2 6362.41 s | 两卡在该图上仅带来约 1.10×；旧版 OptRoll 没有优于 FIFO，单纯调度的收益受到限制 | 10 候选、3/3/2/2 探索性重复，未执行完整 35 轮或正式四系统比较；不是论文级最终性能结果 |

来源：[A100 Phase1 报告](../experiments/a100-original-baseline/reports/final_reproduction.md)、[A100 Phase2 报告](../experiments/a100-original-baseline/reports/phase3/phase2_terminal_partial.md)、[H100 pilot](../experiments/h100-rollout-pilot/reports/ADAPTIVE-PILOT-REPORT.md)。请求时间与 worker wall 的定义不同，worker wall 也不等于完整的 Agent 决策延迟。

## 目前可以作为论文 observation 的四点

1. **推理加速与候选完成速度可能方向相反。** R9 的五次生成请求快了 1.613×，但在现有冷进程适配器下，worker wall 相比 dense 增加约 35.4%。这是同一组实测边界内的观察，提示编译、加载、预热等成本应进入 rollout 目标。不能把这个例子扩大成所有候选或硬件的一般规律。
2. **暴露更多卡并不能保证 Agent 使用更多卡。** Phase1 六小时内只出现 0/1/2 卡进程分配，GPU2–7 没有被分配；Phase2 只分配了 GPU0。我们已测得资源空余，但“多少空余可由调度消除”仍需有同一候选图的对照实验。
3. **依赖和验收流程会限制并发收益。** H100 pilot 中，FIFO2 与旧 OptRoll2 的 TTVF 相差约 0.4%，没有方法优势证据。所有有效运行的候选语义判断一致，按中位数选出的 frontier 一致，但两个单次原始 frontier 不同。这个负结果支持继续研究持久模型 worker、参考结果复用和关键路径；这些改进的收益尚未测得。
4. **可信验收本身是 rollout 的一部分。** 实验中出现过子评估失败却保留旧 LPIPS 分数，以及 Agent 没有真正收到图片却宣称已看图的问题。远端退出码、输入/输出哈希、实际图片输入和持久化轨迹让问题可被发现并修复。这里观察到的是“原版收集逻辑 + 缺依赖的站点适配”中的失败路径，不能声称原版在完整环境下必然失败。[复验时间线](../experiments/a100-original-baseline/reports/phase3/master_reverify_recovery_timeline.md)

## 八卡利用率的准确口径

| 独立窗口 | 8 卡容量 | 进程分配 GPU-s | 分配占比 | NVML 利用率估计 |
| --- | ---: | ---: | ---: | ---: |
| 2026-09-13 Phase1，6 h | 172800 GPU-s | 20577.597 | 11.91% | 3.48%，按 1 Hz 样本近似 |
| 2026-09-14 Phase2，6 h | 172800 GPU-s | 5126.073 | 2.97% | 1.64%–1.76%，显式计入缺失区间范围 |

进程分配不是 GPU 实际计算时间，NVML 也不是精确 SM occupancy。Phase1 采用每个样本约 1 秒的近似，Phase2 用同机 monotonic 时间加权并把 25.142 s 缺口标为未知。两个窗口之间有未计入上述分母的 CPU 续跑，不能把两期百分比当成一次连续完整实验的利用率。原版搜索决策、质量依赖、站点修复以及截止后的等待共同影响空闲；低利用率不能全部算作原版调度缺陷。

数据：[Phase1](../experiments/a100-original-baseline/reports/full_gpu_window_six_hour_cap.json)、[Phase2](../experiments/a100-original-baseline/artifacts/phase3/phase2_full_window_terminal/full_window_cap.json)。

## 质量和复现范围

A100 Cache R05/R08 经过新的原版 LPIPS 计算和 Master 抽样画面检查。每点在 5 个 prompts 中选择 42/80 对导出帧计算 LPIPS，max 分别为 0.0793335 和 0.0260914。Master 实际接收了 20 张对照 overview/detail 图。连续播放、未抽到的帧和更广泛 prompts/seeds 没有覆盖。原始 `nvidia_gemini:api_key_missing` 标记保留，lite 路径之外的独立 graph reviewer 未执行。不能写成“广泛无损”“完整论文复现”或“全部质量门通过”。

Seq0002 没有在可见轨迹里读取旧候选结果，但允许源码包含原有优化开关和速度提示，所以“绝对没有任何历史经验的盲搜索”也不成立。此前 seq0001 的污染尝试被单独排除，失败与不合格候选保留在报告中。[方法边界](../experiments/a100-original-baseline/reports/methodology_boundaries.md)

## 还缺什么

目前材料适合写论文的 problem、measurement 和 observation 部分。要形成 ICML/CVPR 论文中的方法贡献，还需要以下直接证据：

- 完成 A100 Kernel+Cache 组合及其自身独立验收，封存整个 baseline 的交付；不能用之前的 H100 两-prompt 组合替代。
- 对同一候选集合比较改进后的 runtime、现有 OptRoll、严格 FIFO 和 work-conserving FIFO，区分持久 worker/结果复用与纯调度收益，报告 decision agreement 与 TTVF。
- 对加载、编译复用、参考评分复用、调度分别做消融，并把站点修复时间单列；先沿用小 pilot 检验收益，再扩大 trace。
- 增加 prompts/seeds 和质量评估范围；当前 A100 单 seed 的 5 prompts 与 H100 pilot 的质量合同不同。
- 后续单独评测跨硬件/模型 memory 与新 kernel/方法检索，包括迁移收益、失效经验和检索贡献；这两项目前均为 NOT_RUN。

另一项 8×A100 FIFO8/OptRoll8 pilot 的本地最后记录仍是 2026-09-12 的运行中 checkpoint。此次远端旧入口未能连接，不能确认最终配对比较，因此本总结没有把它纳入性能结论。

可以使用的暂定论文主张是：**在这些视频优化 Agent 轨迹中，生成请求延迟不足以描述 rollout 效率；冷启动成本、依赖链和验收证据共同决定候选何时可信地完成。** “我们的系统已解决这些问题并取得显著提升”仍需后续对照实验支持。
