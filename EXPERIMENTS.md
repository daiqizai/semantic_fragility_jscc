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

# GPU 约束

训练、训练型 dry-run 和长时间实验只能使用物理 GPU 4、5、6、7，禁止使用
物理 GPU 0、1、2、3。命令必须通过 `CUDA_VISIBLE_DEVICES` 显式选择设备。
例如 `CUDA_VISIBLE_DEVICES=7 ... --device cuda:0` 中的 `cuda:0` 实际对应
物理 GPU 7。

# 总索引

| 实验 ID | 状态 | 有效性 | 日期 | 目的 | 结果摘要 | 配置与产物 |
|---|---|---|---|---|---|---|
| EXP-S0-001 | DONE | N/A | 2026-06-12 | GPU 训练、追踪、checkpoint 与 LPIPS dry-run | RTX 4090 上完整通过；仅为工程验证，不支撑研究结论 | `configs/EXP-S0-001_gpu_dryrun.json`、`outputs/EXP-S0-001/` |
| EXP-S1-001 | INVALID | N/A | 2026-06-12 | CIFAR-10 ResNet-18 分类器 baseline | 沙箱内 CUDA 不可见，训练开始前失败 | `configs/EXP-S1-001_classifier.json`、`outputs/EXP-S1-001/` |
| EXP-S1-002 | INVALID | N/A | 2026-06-12 | 旧 CIFAR-10 AWGN DeepJSCC 计划 | 从未运行；缺少完整 test 评估，由 EXP-S1-003 替代 | `configs/EXP-S1-002_deepjscc.json` |
| EXP-S1-003 | INVALID | N/A | 2026-06-12 | 旧 CIFAR-10 AWGN DeepJSCC 计划 | 从未运行；分类器依赖失效，由 EXP-S1-005 替代 | `configs/EXP-S1-003_deepjscc.json` |
| EXP-S1-004 | DONE | FRESH | 2026-06-12 | CIFAR-10 ResNet-18 分类器 baseline 重跑 | 最佳 test accuracy 95.29%，独立 checkpoint 复算一致 | `configs/EXP-S1-004_classifier.json`、`outputs/EXP-S1-004/` |
| EXP-S1-005 | DONE | FRESH | 2026-06-13 | CIFAR-10 AWGN DeepJSCC baseline | CBR 1/3；0–20 dB PSNR 22.27–32.42 dB，重建准确率 62.79%–91.33% | `configs/EXP-S1-005_deepjscc.json`、`outputs/EXP-S1-005/` |
| EXP-S2-001 | INVALID | N/A | 2026-06-12 | 旧四种排序 held-out Top-K 计划 | 从未运行；checkpoint 依赖失效，由 EXP-S2-002 替代 | `configs/EXP-S2-001_ranking.json` |
| EXP-S2-002 | TODO | N/A | - | 四种排序的 held-out Top-K 验证 | 两个阶段1 checkpoint 已就绪 | `configs/EXP-S2-002_ranking.json` |

# EXP-S0-001

- 状态：`DONE`
- 有效性：`N/A`（工程 dry-run，不作为论文实验）
- 日期：2026-06-12
- Git commit：`c65220d4e347ce5bf8a9508e28f22319bbc7a116`
- Git 工作区：不干净；完整状态记录于
  `outputs/EXP-S0-001/run_manifest.json`
- 数据集：CIFAR-10 官方 train/test split，各读取首个 8 样本 batch
- 模型：随机初始化 CIFAR ResNet-18 与 ConvDeepJSCC，各执行一个优化步骤
- 信道：实值 AWGN，5 dB
- 配置：`configs/EXP-S0-001_gpu_dryrun.json`
- 随机种子：7
- 命令：

```bash
CUDA_VISIBLE_DEVICES=7 \
/data2/liulu/miniconda3/envs/semantic/bin/python \
  scripts/gpu_dry_run.py \
  --config configs/EXP-S0-001_gpu_dryrun.json \
  --device cuda:0
```

- 环境：NVIDIA GeForce RTX 4090；PyTorch `2.7.1+cu118`；CUDA `11.8`
- 日志：`outputs/EXP-S0-001/run.log`
- 汇总：`outputs/EXP-S0-001/summary.json`
- Checkpoint：
  - `checkpoints/EXP-S0-001/classifier_step.pt`
  - `checkpoints/EXP-S0-001/jscc_step.pt`
- 验证结果：分类器与 DeepJSCC 前向、反向和优化步骤通过；两个 checkpoint
  可重新加载；真实 AlexNet LPIPS 与完整单批质量/语义指标通过；峰值 CUDA
  已分配内存 `180,979,712` bytes。
- 状态解释：仅验证 CUDA、数据加载、日志、manifest、checkpoint 和 LPIPS
  流程。随机初始化单批指标不具有研究意义，不进入结果表。

# EXP-S1-001

- 状态：`INVALID`
- 有效性：`N/A`
- 日期：2026-06-12
- Git commit：`6822ca1919fb34765108014750cbc0769bf9afdf`
- Git 工作区：干净
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
- 状态解释：沙箱内启动后，模型迁移到 CUDA 时报告
  `RuntimeError: No CUDA GPUs are available`，尚未执行训练 step，未生成
  checkpoint。失败 manifest 保存在 `outputs/EXP-S1-001/run_manifest.json`；
  沙箱外验证同一环境可访问物理 GPU 7。按 ID 不可复用规则由
  `EXP-S1-004` 重跑。

# EXP-S1-004

- 状态：`DONE`
- 有效性：`FRESH`
- 日期：2026-06-12
- Git commit：`7ab8259e6b72ceee986aef0cfa0ecb3571a23c83`
- Git 工作区：干净
- 数据集：CIFAR-10，官方 train/test split
- 模型：CIFAR ResNet-18
- Checkpoint：`checkpoints/EXP-S1-004/best.pt`
- 配置：`configs/EXP-S1-004_classifier.json`
- 随机种子：7
- 命令：

```bash
CUDA_VISIBLE_DEVICES=7 \
/data2/liulu/miniconda3/envs/semantic/bin/python \
  scripts/train_classifier.py \
  --config configs/EXP-S1-004_classifier.json \
  --device cuda:0
```

- 日志：`outputs/EXP-S1-004/run.log`
- 指标：最佳 test accuracy `0.9529`（epoch 95）；epoch 100 test
  accuracy `0.9519`；最终 train cross-entropy `0.0024217`
- 汇总：`outputs/EXP-S1-004/summary.json`
- Manifest：`outputs/EXP-S1-004/run_manifest.json`
- 环境：NVIDIA GeForce RTX 4090；物理 GPU 7；PyTorch `2.7.1+cu118`
- 运行时间：2026-06-12 17:34:18 至 17:47:59 UTC
- 验证结果：`best.pt` 在 CPU 独立重载时无 missing/unexpected key；对官方
  10,000 张 test split 复算得到 `9529/10000 = 0.9529`。
- 状态解释：正式分类器 baseline 完成，可作为后续 DeepJSCC 语义评估模型。

# EXP-S1-002

- 状态：`INVALID`
- 有效性：`N/A`
- 日期：2026-06-12
- Git commit：N/A（从未运行）
- Git 工作区：N/A（从未运行）
- 数据集：CIFAR-10，官方 train split
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
- 状态解释：从未启动。该计划只记录 train MSE，不能满足阶段1 baseline
  评估要求；按不可复用规则由 `EXP-S1-003` 替代。

# EXP-S1-003

- 状态：`INVALID`
- 有效性：`N/A`
- 日期：2026-06-12
- Git commit：N/A（从未运行）
- Git 工作区：N/A（从未运行）
- 数据集：CIFAR-10，官方 train/test split
- 模型：ConvDeepJSCC，latent channels = 16
- 信道：实值 AWGN
- 训练 SNR：均匀采样 `[0, 20]` dB
- 测试 SNR：0、5、10、15、20 dB
- Checkpoint：`checkpoints/EXP-S1-003/latest.pt`
- 分类器：`checkpoints/EXP-S1-001/best.pt`
- 配置：`configs/EXP-S1-003_deepjscc.json`
- 随机种子：训练 7；所有测试 SNR 使用配对信道种子 1007
- 指标：MSE、PSNR、四尺度 CIFAR MS-SSIM、LPIPS、实值 CBR、clean
  accuracy、reconstruction accuracy、prediction consistency、semantic
  failure rate、semantic KL
- 命令：

```bash
CUDA_VISIBLE_DEVICES=7 \
/data2/liulu/miniconda3/envs/semantic/bin/python \
  scripts/train_jscc.py \
  --config configs/EXP-S1-003_deepjscc.json \
  --device cuda:0
```

- 日志：`outputs/EXP-S1-003/run.log`
- 指标输出：`outputs/EXP-S1-003/metrics.jsonl`
- 汇总：`outputs/EXP-S1-003/summary.json`
- 状态解释：从未启动。原计划依赖已失败且不可复用的 `EXP-S1-001`
  checkpoint，因此由 `EXP-S1-005` 替代。

# EXP-S1-005

- 状态：`DONE`
- 有效性：`FRESH`
- 日期：2026-06-13
- Git commit：`40d89a99b3eb6db5a2d1c048379c4eea207535b0`
- Git 工作区：干净
- 数据集：CIFAR-10，官方 train/test split
- 模型：ConvDeepJSCC，latent channels = 16
- 信道：实值 AWGN
- 训练 SNR：均匀采样 `[0, 20]` dB
- 测试 SNR：0、5、10、15、20 dB
- Checkpoint：`checkpoints/EXP-S1-005/latest.pt`
- 分类器：`checkpoints/EXP-S1-004/best.pt`
- 配置：`configs/EXP-S1-005_deepjscc.json`
- 随机种子：训练 7；所有测试 SNR 使用配对信道种子 1007
- 指标：MSE、PSNR、四尺度 CIFAR MS-SSIM、LPIPS、实值 CBR、clean
  accuracy、reconstruction accuracy、prediction consistency、semantic
  failure rate、semantic KL
- 命令：

```bash
CUDA_VISIBLE_DEVICES=7 \
/data2/liulu/miniconda3/envs/semantic/bin/python \
  scripts/train_jscc.py \
  --config configs/EXP-S1-005_deepjscc.json \
  --device cuda:0
```

- 日志：`outputs/EXP-S1-005/run.log`
- 指标输出：`outputs/EXP-S1-005/metrics.jsonl`
- 汇总：`outputs/EXP-S1-005/summary.json`
- Manifest：`outputs/EXP-S1-005/run_manifest.json`
- 环境：NVIDIA GeForce RTX 4090；物理 GPU 7；PyTorch `2.7.1+cu118`
- 运行时间：2026-06-13 06:04:34 至 06:10:13 UTC
- 训练结果：100 epoch 完成，最终 train MSE `0.0017085885`；checkpoint
  `latest.pt` 记录 epoch 100，CPU 独立重载时无 missing/unexpected key。
- 测试结果：

| SNR (dB) | PSNR (dB) | MS-SSIM | LPIPS | 重建准确率 | Consistency | Failure rate | Semantic KL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 22.2670 | 0.862133 | 0.032070 | 0.6279 | 0.6306 | 0.3378 | 1.453399 |
| 5 | 26.2265 | 0.939615 | 0.013055 | 0.8208 | 0.8319 | 0.1441 | 0.575251 |
| 10 | 29.3432 | 0.970882 | 0.006119 | 0.8840 | 0.9018 | 0.0788 | 0.286955 |
| 15 | 31.4062 | 0.982888 | 0.003749 | 0.9085 | 0.9280 | 0.0544 | 0.200653 |
| 20 | 32.4223 | 0.987116 | 0.002986 | 0.9133 | 0.9333 | 0.0496 | 0.175833 |

- 共同测试条件：每个 SNR 均使用完整 10,000 张 test split；clean accuracy
  `0.9529`；实值 CBR `1/3`；测试信道种子 `1007`。
- 状态解释：正式 DeepJSCC baseline 完成，阶段1两个 checkpoint 均已就绪；
  图像质量与语义指标随 SNR 整体改善，可进入阶段2 held-out 排序验证。

# EXP-S2-001

- 状态：`INVALID`
- 有效性：`N/A`
- 日期：2026-06-12
- Git commit：N/A（从未运行）
- Git 工作区：N/A（从未运行）
- 数据集：CIFAR-10 官方 test split，最多 1,000 张
- 模型：冻结 DeepJSCC 与 ResNet-18
- Checkpoint：
  - `checkpoints/EXP-S1-001/best.pt`
  - `checkpoints/EXP-S1-003/latest.pt`
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
- 状态解释：从未启动。依赖的两个阶段1 checkpoint ID 已失效，由
  `EXP-S2-002` 替代。

# EXP-S2-002

- 状态：`TODO`
- 有效性：`N/A`
- 日期：尚未运行
- Git commit：运行时由 manifest 记录
- Git 工作区：运行时必须干净
- 数据集：CIFAR-10 官方 test split，最多 1,000 张
- 模型：冻结 DeepJSCC 与 ResNet-18
- Checkpoint：
  - `checkpoints/EXP-S1-004/best.pt`
  - `checkpoints/EXP-S1-005/latest.pt`
- 信道：AWGN
- 测试 SNR：0、5、10 dB
- 排序：random、activation saliency、gradient x activation、oracle fragility
- 粒度：channel group，group size = 2
- Oracle Monte Carlo：4
- Top-K 比例：0.1、0.25、0.5
- 配置：`configs/EXP-S2-002_ranking.json`
- 随机种子：7
- 命令：

```bash
CUDA_VISIBLE_DEVICES=7 \
/data2/liulu/miniconda3/envs/semantic/bin/python \
  scripts/run_ranking.py \
  --config configs/EXP-S2-002_ranking.json \
  --device cuda:0
```

- 日志：`outputs/EXP-S2-002/run.log`
- 结果：`outputs/EXP-S2-002/ranking_results.json`
- 指标：尚无
- 状态解释：`EXP-S1-004` 与 `EXP-S1-005` checkpoint 均已就绪，可以启动。

# 结果表

没有实际运行的实验不得填数值。

| 实验 ID | Git commit | 数据集/划分 | 模型 | 信道 | 训练 SNR | 测试 SNR | CBR | Seed | PSNR | MS-SSIM | LPIPS | Accuracy | Consistency | Failure rate | 配置/日志/Checkpoint |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| EXP-S1-004 | `7ab8259` | CIFAR-10 official train/test | CIFAR ResNet-18 | N/A | N/A | N/A | N/A | 7 | N/A | N/A | N/A | 0.9529 | N/A | N/A | `configs/EXP-S1-004_classifier.json` / `outputs/EXP-S1-004/` / `checkpoints/EXP-S1-004/best.pt` |
| EXP-S1-005 | `40d89a9` | CIFAR-10 official train/test | ConvDeepJSCC, latent 16 | real AWGN | uniform [0, 20] dB | 0 dB | 1/3 | 7 | 22.2670 | 0.862133 | 0.032070 | 0.6279 | 0.6306 | 0.3378 | `configs/EXP-S1-005_deepjscc.json` / `outputs/EXP-S1-005/` / `checkpoints/EXP-S1-005/latest.pt` |
| EXP-S1-005 | `40d89a9` | CIFAR-10 official train/test | ConvDeepJSCC, latent 16 | real AWGN | uniform [0, 20] dB | 5 dB | 1/3 | 7 | 26.2265 | 0.939615 | 0.013055 | 0.8208 | 0.8319 | 0.1441 | same as above |
| EXP-S1-005 | `40d89a9` | CIFAR-10 official train/test | ConvDeepJSCC, latent 16 | real AWGN | uniform [0, 20] dB | 10 dB | 1/3 | 7 | 29.3432 | 0.970882 | 0.006119 | 0.8840 | 0.9018 | 0.0788 | same as above |
| EXP-S1-005 | `40d89a9` | CIFAR-10 official train/test | ConvDeepJSCC, latent 16 | real AWGN | uniform [0, 20] dB | 15 dB | 1/3 | 7 | 31.4062 | 0.982888 | 0.003749 | 0.9085 | 0.9280 | 0.0544 | same as above |
| EXP-S1-005 | `40d89a9` | CIFAR-10 official train/test | ConvDeepJSCC, latent 16 | real AWGN | uniform [0, 20] dB | 20 dB | 1/3 | 7 | 32.4223 | 0.987116 | 0.002986 | 0.9133 | 0.9333 | 0.0496 | same as above |

# STALE 记录

当前没有需要标为 `STALE` 的实验。
