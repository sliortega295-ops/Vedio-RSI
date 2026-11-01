# Vedio-RSI：实验结果与下一步方向

最新整理：**2026-10-31**。本次发布汇总截至 10 月 24 日的实验/搜索证据及 10 月 26 日的 ResearchBoost 源码审阅，不代表 10 月 31 日新跑了实验。

先看两份文档：

- [目前做到了什么、哪些结论还不能说](reports/RESEARCH-STATUS-20261031.md)
- [下一步优先级与验收条件](reports/NEXT-STEPS-20261031.md)

核心判断：完整优化流程已经在 SANA 和 Wan 的站点适配版上走通，但“流程结束”“质量通过”“组合成功”和“论文结果复现”不是一回事。SANA 六类搜索得到 **1.2933×** 的已验收组合；Wan 得到独立通过的组件，但两个组合都被质量 gate 拒绝。搜索调试扩展到 **39 个方法/组合 + 2 个引擎**，仍不是 41 个实测成功方法。

| 证据 | 阅读入口 | 需要保留的边界 |
|---|---|---|
| A/B/C/D 完整运行 | [报告](evidence/20261031/abcd/REPORT.md)、[汇总](evidence/20261031/abcd/SUMMARY.json) | 每组 55 个视频；自适应搜索结果不同，不能直接归因调度收益 |
| DAG 首轮 | [报告](evidence/20261031/dag/REPORT.md)、[有效性](evidence/20261031/dag/CAMPAIGN-VALIDITY.json) | `INVALID_INFRA_TRIAL`，不是成功的受控对照 |
| SANA 六类搜索 | [报告](evidence/20261031/sana/REPORT.md)、[交付](evidence/20261031/sana/INTEGRATED-DELIVERY.json) | 1.2933×；五 prompt、单 seed；有基础设施介入 |
| 旧 Cache 复核 | [报告](evidence/20261031/quality-bridge/REPORT.md)、[争议通知](evidence/20261031/quality-bridge/QUALITY-CONFLICT-NOTICE.json) | 产物可复现，原质量 pass 与匿名复评 fail 并存 |
| Wan 六类搜索 | [报告](evidence/20261031/wan/REPORT.md)、[交付](evidence/20261031/wan/INTEGRATED-DELIVERY.json) | `NO_VALID_FRONTIER` 指无可接受组合；不是所有组件都失败 |
| Sparse 搜索调试 | [目录报告](evidence/20261031/search/REPORT.md)、[CSV](evidence/20261031/search/methods.csv) | 主控静态核查；GPU 验证 0、经验入库 0 |
| ResearchBoost 借鉴 | [源码审阅与边界](reports/RESEARCHBOOST-REVIEW-20261026.md) | 固定版本分析，不导入第三方代码，不声称复现实验 |

这是一条**结果与证据发布 PR**，不是最新运行时源码的完整发布。原始视频、权重、完整日志和未提交的实验代码没有混入；公开证据也不等价于可独立重跑的全量 artifact。详见 [发布说明](PUBLICATION-NOTE.md)、[来源与哈希清单](PUBLICATION-MANIFEST.json)。

历史工作保留不动：[PR #1：10 月 1 日基线快照](https://github.com/sliortega295-ops/Vedio-RSI/pull/1)；[PR #2：FastRoll 实现](https://github.com/sliortega295-ops/Vedio-RSI/pull/2)。#2 中的 31 项 CPU 测试与 GPU `NOT_RUN`，不能被本次其他分支上的 GPU 结果替换。

离线检查发布包：

```bash
python3 -B tools/verify_status.py
```
