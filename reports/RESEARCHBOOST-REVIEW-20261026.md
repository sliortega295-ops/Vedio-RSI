# ResearchBoost-Agent：值得借鉴的约束，不是现成的研究结论

这是 2026-10-26 审阅的固定版本摘要，2026-10-31 整理发布。对象为 [官方仓库](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/tree/5126f0cd0537fe9bb00049aa1f162c01e3d66dc2)，commit `5126f0cd0537fe9bb00049aa1f162c01e3d66dc2`；不是对后续 HEAD 的持续审计。本次没有运行其 GPU 实验或导入其代码。

## 最值得学的三个设计

1. **优化者不能自己宣布成功。** 最终外部验收检查完整 check 的 PASS、最低速度目标和启动时冻结的 judge 摘要；不是看到 Agent 写了“完成”就接受。[accept.py](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/5126f0cd0537fe9bb00049aa1f162c01e3d66dc2/goal/clawo/accept.py)
2. **先冻结正确性与性能规则。** Judge 包含结构约束、baseline/candidate/baseline 计时、逐位与成对 seed 统计路径，并用已知坏例检查规则能否拒绝违规改动。我们需要借鉴“规则版本与证据不可偷换”，不能把训练任务的数值阈值直接照搬到视频质量。[judge 设计](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/5126f0cd0537fe9bb00049aa1f162c01e3d66dc2/docs/judge.md)
3. **经验写成适用条件与实际执行检查。** 优化动作需检查目标路径是否真的激活、是否落入 fallback；长会话由外围恢复机制继续推进。这比简单积累“某个技巧很好”更适合可审计经验。[动作技能](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/5126f0cd0537fe9bb00049aa1f162c01e3d66dc2/skills/train-opt-action-ladder/SKILL.md)、[Goal mode](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent/blob/5126f0cd0537fe9bb00049aa1f162c01e3d66dc2/docs/goal-mode.md)

## 对我们意味着什么

源码公开说明这不是只有宣传页；但不能据此说它已验证开放论文检索、跨模型自动 Memory 或我们的八卡依赖式 rollout。作者不同项目的速度数字有不同硬件、精度、任务和测量范围，也不是我们复现的证据。本发布包不重新制作横向排名表。

我们最直接的改进是：冻结视频质量判定合同、记录版本与争议；区分组件成功/组合失败；对稀疏和量化方法记录真实 kernel 激活与 fallback；以外部证据批准经验准入。它不能替代当前缺失的 Wan 方法筛选与 GPU 验证。

## 代码复用边界

在上述审阅快照中未发现仓库级明确 LICENSE，因此这里只提供设计分析与固定版本链接，不复制第三方实现。“源码能看”不自动等于已确认可再分发授权。实际采用前应重新核对当前许可证或向作者确认；这不是对后续仓库状态的断言。
