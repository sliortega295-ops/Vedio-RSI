> 2026-10-31 发布副本：保留原报告的实验日期、结论和历史状态；私有路径已替换，未发布产物不伪装为可访问链接。文末“未推送”等措辞指报告生成时；本次发布说明见根目录 PUBLICATION-NOTE.md。

# Sparse attention 搜索模块单独调试

状态：SEARCH_ONLY_COMPLETED；GPU、模型下载、候选生成、质量评测、经验入库均未运行。

## 这轮得到了什么

完成主目录 41 项去重研究的静态核验：39 项方法/方法组合，2 项执行引擎。
其中 28 项找到相应实现源文件，3 项所查分支只有占位或展示材料，2 项未绑定论文版本实现，8 项本轮尚未定位实现。
另保留 15 条论文级新线索等待代码审查；不把它们混进上述 41 项，也不宣称已经穷尽领域。
有源文件 ≠ 已确认能编译 ≠ 在 Wan1.3B/A100 上有效 ≠ 保证质量。28 项也不代表 28 个独立仓库。

本轮由主控按照搜索策略执行检索、审阅原始来源；不是新会话自主 agent 的无提示召回率实验。
它证明旧搜索漏掉了大量可调查的方法，并验证检索/证据整理工具；不能据此宣称搜索成功率或最优质量得到提高。

## 以前为什么只剩下几个

历史 EXPORT 中 sparse 共 5 条查询，均围绕 Sparse-VideoGen、SpargeAttn、Sliding Tile Attention。每条 limit=3。
首条将两个不同方法名串在同一查询中，返回零；最后一条失败。中间三条总共暴露 7 个来源条目。
这是已知名字反复查证，不是系统探索。整个历史运行查询 40 条，不能把漏检全部归因于总预算耗尽。
旧规范 discover≥3、inspect≥2 只是最低覆盖线，不应该被当成检索已经充分的证据。
另外发现并修复来源容量边界 bug：来源池满时检索窗口会缩成 1，排在后面的已知来源无法被补充分类。这是额外回归修复，未证明它是历史漏检主因。

## 已改动的搜索模块

1. 增加独立 search-only 入口：按机制、模型、论文、框架拓展，不向 GPU 执行器提交任务。
2. 初始查询不硬编码 SVG/Sparge 等方法名；从稀疏块、聚类、routing、时间稀疏、校准等角度交叉查询。
3. GitHub/arXiv 支持分页，宽检索窗口为 50；满页仍有后续时明确标记未穷尽，而非暗示没有其他方法。
4. 框架与索引只作为发现入口，论文 ID 负责去重；一个框架多篇论文可以拆开，fork 和改名不能重复计数。
5. 发现、实现存在、当前任务适用性分开；Hopper 限制、训练要求、架构不符不会从发现目录中抹掉方法。
6. 核验引用确实存在于抓取内容、实现路径确实存在于固定 commit 的树；README 不能充当实现代码。
7. 搜索过程保留待审查项；本轮没有修改 formal rollout 预算、搜索轨迹或质量标准，没有写入成功经验。

边界：新增工具负责生成查询、读取单页、展开索引、审计证据；跨来源语义筛选仍由原生模型/主控判断。没有新增偷偷运行的自主 crawler，也没有把人工整理目录当成独立新 agent 的结果。

## 目前最值得继续静态审查的方向

| 方向 | 实现证据 | 当前需要确认的事 |
|---|---|---|
| DraftAttention | 作者仓库有 draft_attention.py；A100/90% 稀疏示例 | 当前 Wan 版本、CFG 两分支与 block-sparse 依赖 |
| PSA | 作者仓库有 PSA kernel 与 Wan1.3B 入口 | 选择不改步数/权重的路径，不能混入 TDM |
| SVOO | 作者仓库有 Wan1.3B/40GB 说明及 attention 源码 | 校准成本、当前版本中的 EAR 组合开关 |
| AdaCluster | 无 README 但有 runwan/wan/modules/model_kvclus.py | 包装接口、聚类成本及依赖版本 |
| DFSAttn / VecAttention | 作者给出 A100 构建说明与 Wan14B 代码 | 1.3B 适配、mask/阈值与双 CFG 分支处理 |
| SPADE | Wan 入口、A100 架构定义、原生稀疏内核 | 具体 dispatch 是否可用及冷启动构建开销 |
| SpargeAttention | Ampere 实现路线已存在 | 区分纯稀疏与稀疏+量化，保持测量规范 |

这些是下一步读代码的优先级，不是“8 个方案已通过”。稀疏度 90%/95% 不等于整体快 10×/20×，也不意味着本任务质量能过关。

## 41 项已核验目录

| 方法/系统 | 训练条件 | 代码状态 | 当前 Wan/A100 判断 |
|---|---|---|---|
| [Sparse VideoGen 1](https://arxiv.org/abs/2502.01776) | 免训练 | 有源文件 | 需移植/依赖核对 |
| [Sparse VideoGen 2](https://arxiv.org/abs/2505.18875) | 免训练 | 有源文件 | 需移植/依赖核对 |
| [SVG-EAR](https://arxiv.org/abs/2603.08982) | 免训练 | 有源文件 | 需移植/依赖核对 |
| [SpargeAttention](https://arxiv.org/abs/2502.18137) | 免训练 | 有源文件 | 优先静态候选，未运行 |
| [SpargeAttention2](https://arxiv.org/abs/2602.13515) | 需要训练 | 版本实现未绑定 | 需不同权重或训练 |
| [Sliding Tile Attention (STA)](https://arxiv.org/abs/2502.04507) | 免训练 | 有源文件 | 需移植/依赖核对 |
| [VSA](https://arxiv.org/abs/2505.13389) | 需要训练 | 有源文件 | 需不同权重或训练 |
| [VSA2](https://arxiv.org/abs/2609.32882) | 需要训练 | 本轮未定位实现 | 需不同权重或训练 |
| [Radial Attention](https://arxiv.org/abs/2506.19852) | 默认免训练/扩长另调 | 有源文件 | 需移植/依赖核对 |
| [DraftAttention](https://arxiv.org/abs/2505.14708) | 免训练 | 有源文件 | 优先静态候选，未运行 |
| [Pyramid Sparse Attention (PSA)](https://arxiv.org/abs/2512.04025) | 免训练路径/可选蒸馏 | 有源文件 | 优先静态候选，未运行 |
| [SVOO](https://arxiv.org/abs/2603.18636) | 免训练但需校准 | 有源文件 | 优先静态候选，未运行 |
| [AdaCluster](https://arxiv.org/abs/2604.18348) | 免训练 | 有源文件 | 优先静态候选，未运行 |
| [DFSAttn](https://arxiv.org/abs/2605.23445) | 免训练 | 有源文件 | 优先静态候选，未运行 |
| [LiteAttention](https://arxiv.org/abs/2511.11062) | 免训练 | 有源文件 | 硬件要求不符 |
| [CalibAtt](https://arxiv.org/abs/2603.05503) | 免训练但需校准 | 有源文件 | 硬件要求不符 |
| [SparsePR](https://arxiv.org/abs/2608.18484) | 免训练 | 有源文件 | 需移植/依赖核对 |
| [VecAttention](https://arxiv.org/abs/2603.29494) | 免训练 | 有源文件 | 优先静态候选，未运行 |
| [HEART](https://arxiv.org/abs/2605.14513) | 免训练 | 本轮未定位实现 | 仅论文级核验 |
| [Compact Attention](https://arxiv.org/abs/2508.12969) | 待进一步核实 | 本轮未定位实现 | 仅论文级核验 |
| [RainFusion](https://arxiv.org/abs/2505.21036) | 免训练 | 本轮未定位实现 | 仅论文级核验 |
| [RainFusion2.0](https://arxiv.org/abs/2512.24086) | 待进一步核实 | 本轮未定位实现 | 仅论文级核验 |
| [AdaSpa](https://arxiv.org/abs/2502.21079) | 免训练 | 本轮未定位实现 | 仅论文级核验 |
| [Sparse-vDiT](https://arxiv.org/abs/2506.03065) | 待进一步核实 | 所查分支占位/材料页 | 实现未发布到所查分支 |
| [QuantSparse](https://arxiv.org/abs/2509.23681) | 待进一步核实 | 所查分支占位/材料页 | 实现未发布到所查分支 |
| [SLA](https://arxiv.org/abs/2509.24006) | 需要训练 | 有源文件 | 需不同权重或训练 |
| [SLA2](https://arxiv.org/abs/2602.12675) | 需要训练 | 版本实现未绑定 | 需不同权重或训练 |
| [BLADE](https://arxiv.org/abs/2508.10774) | 需要训练 | 有源文件 | 需不同权重或训练 |
| [VORTA](https://arxiv.org/abs/2505.18809) | 训练或专用权重 | 有源文件 | 需不同权重或训练 |
| [Veda](https://arxiv.org/abs/2605.30325) | 需要训练 | 有源文件 | 需不同权重或训练 |
| [FPSAttention](https://arxiv.org/abs/2506.04648) | 需要训练 | 所查分支占位/材料页 | 硬件及权重均有限制 |
| [Light Forcing](https://arxiv.org/abs/2602.04789) | 免训练 | 有源文件 | 不同架构/任务 |
| [Sparse Forcing](https://arxiv.org/abs/2604.21221) | 需要训练 | 本轮未定位实现 | 不同架构/任务 |
| [FAST-AR](https://arxiv.org/abs/2602.01801) | 免训练 | 本轮未定位实现 | 不同架构/任务 |
| [SparSTAR](https://arxiv.org/abs/2608.10519) | 免训练 | 有源文件 | 不同架构/任务 |
| [Prism](https://arxiv.org/abs/2610.05416) | 需要训练 | 有源文件 | 不同架构/任务 |
| [HyperVAttention](https://arxiv.org/abs/2607.03012) | 免训练 | 有源文件 | 需移植/依赖核对 |
| [PISA](https://arxiv.org/abs/2602.01077) | 免训练 | 有源文件 | 需移植/依赖核对 |
| [FlashOmni](https://arxiv.org/abs/2509.25401) | 执行引擎 | 有源文件 | 需移植/依赖核对 |
| [SPADE](https://arxiv.org/abs/2608.03335) | 免训练 | 有源文件 | 优先静态候选，未运行 |
| [XAttention](https://arxiv.org/abs/2503.16428) | 免训练 | 有源文件 | 需移植/依赖核对 |

## 关键排错与来源质量

- HASTE 与 HEART 是同一个 arXiv 身份的改名，不能算两种。
- 视频 Compact Attention 与同名 LLM 仓库不是同一个工作；必须核对论文标题/ID。
- Sparse Forcing 论文是可训练方案，不能照搬二手索引的 training-free 分类。
- FPSAttention 的同名账号仓库只是个人主页；另一个候选仓库只有展示材料，未找到 attention 实现。
- HyperVAttention README 顶部遗留错误 paper ID，BibTeX 与真实论文 ID 一致；不能自动信任第一个链接。
- VideoNSA、SwinBERT 和扩散语言模型论文不作为当前视频生成方法计数；DASA 作为 AR KV/特征复用的相邻方向保留。
- STA 本轮绑定的实现来自第三方 SparseVideo，明确标出来源，不伪装成原作者完整复现。
- SparseVideo 是多方法宿主；FastVideo 同样拆成 STA/VSA 等具体论文，不把仓库名当新增优化算法。
- 尚未完成许可证兼容性、安全审计、安装与实测；作者性能数字只作 reported，不写入本地 validated memory。

## 检索范围与未完成的广度

结构化记录 18 条计划查询/分页请求，另有 16 条诊断/按标题查证记录；补充网页批次原始回执单独保存。查询数不是底层 HTTP 总请求数，也不是论文召回率。
计划部分包括 12 条机制网页查询、GitHub 两条初始查询及 3 个后续页面、arXiv 一页 50 条论文。
GitHub framework 查询第三页仍满 50 条，arXiv 第一页也满 50 条。本轮作为有界调试快照结束，没有达到搜索饱和。
网页工具没有给出精确结果总数的查询保留 null，不捏造为 0。检索预算仅是配置上限，不是自动爬虫强制限流器。

新增待查线索如下；已有论文/摘要，但尚未完成与主目录相同深度的代码核验：

- [RECAP: Reciprocal Sparse Attention and Hierarchical Caching for Efficient Video Diffusion](https://arxiv.org/abs/2610.26244)
- [FG-Attn: Leveraging Fine-Grained Sparse Attention in Video Diffusion Models](https://arxiv.org/abs/2509.16518)
- [LVSA: Training-Free Sparse Attention for Long Video Diffusion](https://arxiv.org/abs/2605.31057)
- [Competition Domains Matter: Selective Head Globalization for Training-Free Sparse Attention in Video Diffusion Transformers](https://arxiv.org/abs/2610.14672)
- [LoSA: Near-Lossless Sparse Attention for Training-Free Video Diffusion Acceleration](https://arxiv.org/abs/2608.12032)
- [Mixture of Distributions Matters: Dynamic Sparse Attention for Efficient Video Diffusion Transformers](https://arxiv.org/abs/2601.11641)
- [Bidirectional Sparse Attention for Faster Video Diffusion Training](https://arxiv.org/abs/2509.01085)
- [PARK: Accurate Block Retrieval for Sparse Attention in Video Diffusion Transformers](https://arxiv.org/abs/2609.38978)
- [FVAttn: Adaptive Sparse Attention with Runtime Load Balancing for Video Generation](https://arxiv.org/abs/2607.16190)
- [Input-Aware Sparse Attention for Real-Time Co-Speech Video Generation](https://arxiv.org/abs/2510.02617)
- [Ride the Wave: Precision-Allocated Sparse Attention for Smooth Video Generation](https://arxiv.org/abs/2604.12219)
- [DynamicRad: Content-Adaptive Sparse Attention for Long Video Diffusion](https://arxiv.org/abs/2604.20470)
- [MC-Sparse: Deconstructing and Closing the Dense-Sparse Attention Gap in Diffusion Transformers](https://arxiv.org/abs/2610.06801)
- [WaveAlign: Cache-Aware Query-Row Scheduling for Sparse Attention in Long-Video Generation](https://arxiv.org/abs/2609.34814)
- [Sparsity Should Be ALLOTted by Output: Training-Free Allocation of Block-Sparse Attention for Video Diffusion Transformers](https://arxiv.org/abs/2610.29441)

## 文件与复查

- `reports/AUDIT.json`：机器可读统计与逐项证据绑定。
- `reports/methods.csv`：可筛选的全部目录及适用性备注。
- `reports/papers.csv`：完整论文题名与 arXiv 首次提交年；未通读实验部分，因此不编造论文质量评分。
- `artifacts/catalog.json` / `evidence.json`：人工语义判定与真实抓取内容严格分离。
- `artifacts/repository-trees.json` / `core-files.json`：固定 commit 的代码路径与抽查源码。
- `artifacts/pending-leads.json`：未审查线索及相邻/不适用方向。
- `artifacts/historical-search-audit.json`：旧查询原始证据与源文件哈希。
- `scripts/build_report.py`：从保存的来源和判定重新生成报告，不联网、不启动模型。

测试：9 项新增搜索审计测试 + 99 项现有 method harness 回归测试全部通过（共 108 项）。见 `reports/TEST-RESULTS.json`。

下一步仍先做静态接口/依赖核对，确认“搜到→读代码→适用性分流”完整可审计后，再由用户决定何时恢复 GPU 实验。
