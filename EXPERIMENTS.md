# 实验索引

本文件记录所有正式实验及结果索引。完整逐 epoch 日志、配置快照、指标和图表
保存在对应 `outputs/EXP-xxx/`。当前进度摘要见 `PROGRESS.md`，研究协议见
`PROJECT.md`。

# 状态与有效性

执行状态只使用：

- `TODO`
- `IN_PROGRESS`
- `BLOCKED`
- `DONE`
- `INVALID`

实验结果有效性单独使用：

- `FRESH`：与当前评估协议一致；
- `STALE`：评估逻辑或协议变化后需要重跑；
- `N/A`：尚无结果。

# ID 与目录

实验 ID 使用 `EXP-S{阶段}-{序号}`。失败、中断或配置改变后不得复用旧 ID。

```text
configs/EXP-Sx-NNN_*.json
outputs/EXP-Sx-NNN/config.json
outputs/EXP-Sx-NNN/run.log
outputs/EXP-Sx-NNN/run_manifest.json
outputs/EXP-Sx-NNN/metrics.jsonl
outputs/EXP-Sx-NNN/summary.json
checkpoints/EXP-Sx-NNN/
```

# 总索引

| 实验 ID | 状态 | 有效性 | 日期 | 目的 | 结果摘要 | 配置与产物 |
|---|---|---|---|---|---|---|
| EXP-S1-001 | TODO | N/A | - | CIFAR-10 ResNet-18 分类器 baseline | 尚未运行 | `configs/EXP-S1-001_classifier.json` |
| EXP-S1-002 | TODO | N/A | - | CIFAR-10 AWGN DeepJSCC baseline | 尚未运行；运行前需补完整 test 指标 | `configs/EXP-S1-002_deepjscc.json` |
| EXP-S2-001 | BLOCKED | N/A | - | 四种排序的 held-out Top-K 验证 | 等待 EXP-S1-001 和 DeepJSCC baseline | `configs/EXP-S2-001_ranking.json` |

# EXP-S1-001

- 状态：`TODO`
- 有效性：`N/A`
- 日期：尚未运行
- Git commit：运行时由 manifest 记录
- Git 工作区：运行时必须干净
- 数据集：CIFAR-10，官方 train/test split
- 模型：CIFAR ResNet-18
- Checkpoint：`checkpoints/EXP-S1-001/best.pt`
- 配置：`configs/EXP-S1-001_classifier.json`
- 随机种子：7
- 命令：

```bash
CUDA_VISIBLE_DEVICES=7 \
/data2/liulu/miniconda3/envs/semantic/bin/python \
  scripts/train_classifier.py \
  --config configs/EXP-S1-001_classifier.json \
  --device cuda:0
```

- 日志：`outputs/EXP-S1-001/run.log`
- 指标：尚无
- 状态解释：计划实验，未启动。

# EXP-S1-002

- 状态：`TODO`
- 有效性：`N/A`
- 日期：尚未运行
- Git commit：运行时由 manifest 记录
- Git 工作区：运行时必须干净
- 数据集：CIFAR-10，官方 train split；test 评估尚待实现
- 模型：ConvDeepJSCC，latent channels = 16
- 信道：AWGN
- 训练 SNR：均匀采样 `[0, 20]` dB
- Checkpoint：`checkpoints/EXP-S1-002/latest.pt`
- 配置：`configs/EXP-S1-002_deepjscc.json`
- 随机种子：7
- 原计划命令：

```bash
CUDA_VISIBLE_DEVICES=7 \
/data2/liulu/miniconda3/envs/semantic/bin/python \
  scripts/train_jscc.py \
  --config configs/EXP-S1-002_deepjscc.json \
  --device cuda:0
```

- 日志：`outputs/EXP-S1-002/run.log`
- 指标：尚无
- 状态解释：未启动。当前脚本只记录 train MSE，正式长训练前需增加 test
  PSNR、MS-SSIM、LPIPS、CBR 和语义指标。若配置或评估输出改变，应新建实验 ID，
  不复用 `EXP-S1-002`。

# EXP-S2-001

- 状态：`BLOCKED`
- 有效性：`N/A`
- 日期：尚未运行
- Git commit：运行时由 manifest 记录
- Git 工作区：运行时必须干净
- 数据集：CIFAR-10 官方 test split，最多 1,000 张
- 模型：冻结 DeepJSCC 与 ResNet-18
- Checkpoint：
  - `checkpoints/EXP-S1-001/best.pt`
  - `checkpoints/EXP-S1-002/latest.pt`
- 信道：AWGN
- 测试 SNR：0、5、10 dB
- 排序：random、activation saliency、gradient x activation、oracle fragility
- 粒度：channel group，group size = 2
- Oracle Monte Carlo：4
- Top-K 比例：0.1、0.25、0.5
- 配置：`configs/EXP-S2-001_ranking.json`
- 随机种子：7
- 命令：

```bash
CUDA_VISIBLE_DEVICES=7 \
/data2/liulu/miniconda3/envs/semantic/bin/python \
  scripts/run_ranking.py \
  --config configs/EXP-S2-001_ranking.json \
  --device cuda:0
```

- 日志：`outputs/EXP-S2-001/run.log`
- 结果：`outputs/EXP-S2-001/ranking_results.json`
- 指标：尚无
- 状态解释：缺少两个阶段1 checkpoint，不能运行。

# 结果表

没有实际运行的实验不得填数值。

| 实验 ID | Git commit | 数据集/划分 | 模型 | 信道 | 训练 SNR | 测试 SNR | CBR | Seed | PSNR | MS-SSIM | LPIPS | Accuracy | Consistency | Failure rate | 配置/日志/Checkpoint |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|

# STALE 记录

当前没有正式结果，因此没有需要标为 `STALE` 的实验。

