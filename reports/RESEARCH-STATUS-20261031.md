# 当前研究状态：能闭环，但还没证明开放搜索带来稳定的新收益

整理日期：2026-10-31。实测截止 2026-10-24；不是新一轮实验报告。

## 1. 一句话结论

我们已经有能运行、记录、评估并拒绝坏方案的 Video Optimization Harness；接下来最缺的不是再跑一遍更快的流水线，而是证明：**它能找到 Sol-Agent 固定菜单之外的适用方法，得到可靠的质量收益，并把带条件的经验迁移到后续任务。** 这三件事目前都还没有完整证据。

三条研究线仍是：快速 rollout、开放方法搜索、跨模型/硬件经验。快速 rollout 已有诊断结果；开放搜索有明显漏检后的改进；跨任务 Memory 的收益尚未验证。不能将这些阶段性成果写成整个 RSI 系统已完成。

## 2. 实验时间线与状态

| 实验 | 流程/任务量 | 主要结果 | 应如何使用 |
|---|---|---|---|
| 10/21 A/B/C/D v6 | 各 55 个正式视频；各 4 Kernel、6 Cache、2 Integration 注册 | 四组完整结束；A 4h17m46s，B 4h05m19s，C 3h39m29s，D 3h46m58s | 历史完整流程与资源诊断；不是同一候选轨迹的因果消融 |
| 10/22 DAG v6 | 20 个视频；Cache 的 6 次预算均在接口预检失败 | `INVALID_INFRA_TRIAL`；控制流终止，但没有完成有效 Cache 搜索 | 保留失败，不把 146.52 分钟对 A 的缩短算成 Harness 加速 |
| 10/23 SANA 六类 002 | Kernel 7、Cache 10、Integration 2；60 个实际视频 | 最终 Q/K cast fusion + 保守 EasyCache：1.2933× | 小工作负载上通过当前 gate 的组合，非论文级质量结论 |
| 10/23 Cache bridge | 固定历史配方诊断；15 个新视频、6 组匿名复评 | 旧 Cache 5 个 MP4 字节哈希复现，但原 pass 变为 fail/medium | 质量判断有争议，不是模型生成复现失败 |
| 10/24 Wan full 002 | Kernel 3、Cache 12、Integration 2；65 个视频、11 次模块探测 | 5 个独立组件保留验证证据；2 个组合均拒绝，`NO_VALID_FRONTIER` | 完整负结果闭环，不宣称可用的 2.2× 组合 |
| 10/24 Sparse search-only | 41 个目录身份、另 15 条待代码审查线索 | 28 项发现实现源文件；0 GPU、0 本地方法验收、0 经验入库 | 静态检索诊断，非新 Agent 自主召回实验 |

证据分别见 [ABCD](../evidence/20261031/abcd/REPORT.md)、[DAG](../evidence/20261031/dag/REPORT.md)、[SANA](../evidence/20261031/sana/REPORT.md)、[质量复核](../evidence/20261031/quality-bridge/REPORT.md)、[Wan](../evidence/20261031/wan/REPORT.md)、[搜索](../evidence/20261031/search/REPORT.md)。

“原版”需要限定：A 是恢复的历史 Sol-Agent 控制流程加共同站点适配，不是未经改动的官方软硬件环境；后续 DAG、六类搜索和 Wan port 是修改后的 Harness。它们不能被统称为“原论文完整结果已复现”。10 月 1 日 [PR #1](https://github.com/sliortega295-ops/Vedio-RSI/pull/1) 的 PARTIAL 是当时状态，后续完整运行不应倒改那份历史回执。

## 3. 模型优化到底有多少有效收益

### SANA-Video 2B

六类 002 使用 revision `db5f398b13ca086d09a50ce156c20527773841b1`，832×480、81 帧、50 steps、CFG 6、原五 prompt、seed 42、每视频单卡冷启动与一步预热。

冻结 Dense 的五请求原生耗时合计 **637.2719 s**；最终组合 **492.7363 s，1.2933×**，请求时间减少约 22.68%。50 步不变；没有混入量化或 few-step 的加速。最终 LPIPS mean/max 为 0.020530/0.041007，按原全局规则选出 39 对帧，并完成原生看图与独立验收。[SANA 报告与限制](../evidence/20261031/sana/REPORT.md)

60 个视频中，55 个有可信的完整请求计时；另 5 个生成后遇到 IPC 回传失败。原始聚合 JSON 把产物计为 60 个 completed，不能据此假设 60 个都有可比较耗时。失败占卡约 1.83 GPU-hours 已计入成本，没有排除。

旧 Cache 的跨轮复核针对 **`method-search-20261023-pilot-002`** 的 `easycache_conservative_v1` 与 `integrated_packed_easycache_v1`，不是对 A/B/C/D 全部配方的复审。旧 Cache 重跑 441.993 s、1.448×；新 Kernel 加旧 Cache 374.949 s、1.707×，均被匿名视觉评审拒绝。五个旧 Cache 视频哈希相同，却出现评审分歧，因此保留原 pass 和新 fail，标记 `DISPUTED_QUALITY`。一个新评审者不是人工金标准，也不能用此估计错误率。[争议通知](../evidence/20261031/quality-bridge/QUALITY-CONFLICT-NOTICE.json)

### Wan2.1 T2V 1.3B

原权重 revision `37ec512624d61f7aa208f7ea8140a131f93afc9a`，转换为 Diffusers 布局；1,261 个权重张量的映射核对完成。原生 CLI 与当前适配器的输出对齐、tokenizer runtime parity 仍为 `NOT_RUN`，因此这里是已冻结的 **Wan port baseline**，不是宣称官方原生性能完整复现。[资产审计](../evidence/20261031/wan/ASSET-AUDIT.md)

工作负载为 832×480、81 帧、16 fps、50 steps、CFG 6、五 prompt、seed 42；UniPC flow prediction、flow shift 3；每视频冷启动与一步预热。Dense 五请求合计 **927.5731 s**。

| 条件 | 五请求合计 | 原始速度比 | 验收解释 |
|---|---:|---:|---|
| FastVideo norm fusion 组件 | 约 791.422 s | 1.172× | 独立组件通过；“exact”是该方法检查合同，不是已证明逐位相同 |
| EasyCache 0.03 组件 | 495.301 s | 1.873× | 独立质量检查通过；不是组合或跨 prompt 分布保证 |
| SVG window=2 v2 | 896.830 s | 1.034× | LPIPS mean/max 约 0.516/0.733，质量拒绝 |
| Fusion + EasyCache 0.03 | 420.614 s | 2.205× | 质量拒绝，不是可用加速 |
| Fusion + EasyCache 0.02 | 515.370 s | 1.800× | 质量拒绝，不是可用加速 |

两组合在门框表面产生中等程度的块状纹理/接缝。组件单独通过不保证组合通过。最终 `frontier_points=[]`，但 [VERIFIED-COMPONENTS](../evidence/20261031/wan/VERIFIED-COMPONENTS.json) 中仍有 5 个独立验证组件。[原生交付](../evidence/20261031/wan/INTEGRATED-DELIVERY.json)

机器可读 `NO-FRONTIER-CHECK.json` 的 `issues=[]` 及最终审计 PASS 指负结果交付与证据闭合通过，不能翻译成质量 PASS。受当前根级成功准入规则影响，5 个组件尚未作为成功经验入库；失败轨迹保留。这个粒度问题要修，但不能回填虚假的成功记录。

## 4. GPU 为什么仍然常常空着

| 窗口 | 墙钟 | 视频数 | 八卡平均 NVML | 任务占卡累计 |
|---|---:|---:|---:|---:|
| A | 257.76 min | 55 | 4.84% | 2.049 GPU-h |
| B | 245.32 min | 55 | 4.55% | 1.832 GPU-h |
| C | 219.49 min | 55 | 5.99% | 2.257 GPU-h |
| D | 226.97 min | 55 | 5.47% | 2.156 GPU-h |
| DAG 失败试验 | 146.52 min | 20 | 3.67% | 0.969 GPU-h |
| SANA 六类 002 | 179.36 min | 60（55 个可信计时） | 10.58% | 4.862 GPU-h |
| Wan full 002 | 135.49 min | 65 | 13.98% | 2.669 GPU-h |
| 固定配方 bridge **仅生成窗口** | 5.43 min | 15 | 61.44% | 0.650 GPU-h |

分母不同，最后一行不能作为完整搜索利用率的胜出结果。SANA/Wan 的 NVML 是实际采样服务窗口均值，不覆盖所有主控生命周期；Wan 覆盖 7,249.09 s，完整 8,129.43 s 窗口只能给出 **12.47%–23.30%** 的缺测上下界。NVML 设备活动不是 SM occupancy 或 FLOP 利用率。[SANA 指标](../evidence/20261031/sana/METRICS.json)、[Wan 指标](../evidence/20261031/wan/METRICS.json)

必须分开四种量：

- **模型请求耗时之和**：衡量模型实现，不包含加载、内部预热和视频导出。
- **占卡 GPU-hours**：任务从分配到释放的累计，包含加载、等待及故障占用，不等于有效计算。
- **八卡资源包络**：8 × 墙钟，是可用容量，不是实际使用。Wan 为 18.065 GPU-h；不能称作实际计算成本。
- **完整优化墙钟**：还包括提出方法、实现、评审、独立验收、组合和交付。

Wan 累计请求约 **126.997 GPU-min**，原生生成区间并集约 **27.204 min**，八卡同时处于原生生成约 **7.002 min**；“八个 worker 被占用”另为 10.124 min，不能混用。原生 benchmark 汇总与事件 span 累计相差约 0.002 s，分别保留原口径，不改写历史数值。

Wan API 响应请求并集约 **120.59 min**，累计约 220.87 min（会并发重叠）；SANA 并集约 165.46 min。它们包含模型推理、图片处理、排队、传输和重试，**不是纯网络延迟**。不能直接相减并承诺“去掉 API 就少两小时”。请求体大小也不是 VPN 账单，更不是下载权重的流量。

每秒抽样中，已提交且依赖满足的合法任务没有明显的就绪空卡时间；但 Agent 尚未提交的候选不在这个统计分母里。Wan 仍记录到约 556.5 s 的图级准备机会，不等于全部能安全消除。更准确的诊断是：**任务供应和评审/决策节奏限制并行度，而不只是 GPU 队列派发慢。** 这是观测解释，不是受控因果结论。

## 5. 搜索功能：发现了真正的漏检，但还没转化成新方法收益

Wan 完整实验查了六类，不等于六类都跑了完整视频：

| 类别 | 当前证据 |
|---|---|
| Kernel | FastVideo fusion 有全视频验证；FlashAttention2 因扩展环境未完成这条路线 |
| Cache | TeaCache、EasyCache 有全视频；TaylorSeer 静态检查；已属 Sol 菜单 |
| Sparse attention | SVG 有全视频且失败；不能据此否定其他稀疏方法 |
| Quantization | Sage INT8 QK / FP16 PV 做过 2 次算子探测，无稳定收益；没有 FP8 权重量化的完整视频证据 |
| Token pruning | ToMeSD/VidToMe/PruneVid/ReToMeVA 处于静态适配判断，没有 GPU 方法验证 |
| Few-step | RhymeFlow/DPM 等做来源/代码查阅；当前固定 50 步/权重合同下没有运行 |

旧 sparse 搜索仅有 5 条围绕已知名字的查询、每条 limit=3；并非整个搜索预算已经耗尽。后来的独立搜索调试扩展到 **39 方法/组合 + 2 引擎**；实现状态为 28 有源文件、3 占位/材料页、2 版本未绑定、8 本轮未定位。另 15 条论文线索待代码核查；`pending-leads.json` 有 25 行，其中 10 行属于相邻/排除方向，不应算作 25 个新方法。

目录包含 DraftAttention、PSA、SVOO、AdaCluster、DFSAttn、VecAttention、SPADE、SpargeAttention 等可继续静态审查的路线。VSA 的训练稀疏代码已找到，但模型权重、训练成本和任务适配尚未验证。训练稀疏与 few-step 不应因为不符合当前权重合同就在发现目录消失，也不能直接当成免训练 drop-in。[搜索目录与来源索引](../evidence/20261031/search/REPORT.md)

该 41 项目录由主控审阅产生，**不是**全新 Agent 未受提示的搜索成绩。新策略是否提高自主发现能力、其他五类是否同样拓宽、是否带来 Sol 之外的有效优化，仍为 `NOT_RUN`。

## 6. 目前能支持文章的判断

**有证据的观察：** 视频优化 Agent 的端到端成本不只在视频生成；相同视频可出现质量判定分歧；独立有效组件可能组合失效；窄查询会漏掉大量可调查方法；失败和恢复必须作为正式轨迹保存。

**尚不能作为贡献结论：** 通用调度算法因果加速、自动搜索发现新方法后的稳定收益、跨模型/跨硬件 Memory 减少试错、论文级视频质量保持、无人介入的长期闭环。D 的复用观察不等于跨候选 persistent model 已完成有效比较；搜索目录也不等于经验库。

对 ICML/CVPR 更合理的主线是：**在固定质量约束下，开放方法发现与带适用条件的经验，能否让视频部署优化更可靠、更少试错？** rollout 是支撑效率的方法之一，不单独包装成已完成的系统论文。具体下一步见 [执行优先级](NEXT-STEPS-20261031.md)。
