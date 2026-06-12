# 实验索引

本文件只做完整实验索引。项目当前状态、研究结论、风险与下一步以
[`PROGRESS.md`](PROGRESS.md) 为唯一准绳。任何实验必须先分配唯一 ID，
并使用同名配置、输出目录和 checkpoint 目录。

## ID 规则

- 阶段0：`EXP-S0-NNN`
- 阶段1 Baseline：`EXP-S1-NNN`
- 阶段2 Fragility 可行性验证：`EXP-S2-NNN`
- 阶段3 固定预算保护：`EXP-S3-NNN`
- 阶段4 完整模型：`EXP-S4-NNN`
- 阶段5 完整实验：`EXP-S5-NNN`
- 阶段6 论文整理：`EXP-S6-NNN`

失败或中断的实验 ID 不得复用。修改配置后重跑必须创建新 ID。

## 目录约定

```text
configs/EXP-Sx-NNN_*.json
outputs/EXP-Sx-NNN/config.json
outputs/EXP-Sx-NNN/run.log
outputs/EXP-Sx-NNN/run_manifest.json
outputs/EXP-Sx-NNN/metrics.jsonl
outputs/EXP-Sx-NNN/summary.json
checkpoints/EXP-Sx-NNN/
```

## 实验列表

| 实验 ID | 阶段 | 状态 | 目的 | 配置 | 输出 | Checkpoint | 备注 |
|---|---|---|---|---|---|---|---|
| EXP-S1-001 | 阶段1 | 计划 | 训练 CIFAR-10 ResNet-18 语义分类器 | `configs/EXP-S1-001_classifier.json` | `outputs/EXP-S1-001/` | `checkpoints/EXP-S1-001/best.pt` | 尚未运行 |
| EXP-S1-002 | 阶段1 | 计划 | 训练 CIFAR-10 AWGN DeepJSCC baseline | `configs/EXP-S1-002_deepjscc.json` | `outputs/EXP-S1-002/` | `checkpoints/EXP-S1-002/latest.pt` | 尚未运行 |
| EXP-S2-001 | 阶段2 | 阻塞 | 比较四种排序方法的 held-out Top-K 破坏效果 | `configs/EXP-S2-001_ranking.json` | `outputs/EXP-S2-001/` | 无新权重 | 等待 EXP-S1-001、EXP-S1-002 |

## 失效结果

目前没有正式实验结果，因此没有因评估代码修改而需要重跑的旧结果。

