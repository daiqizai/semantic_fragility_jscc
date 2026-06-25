# 当前阶段

**阶段2：Fragility 可行性验证**

状态：`IN_PROGRESS`

阶段0准备与定义、阶段1分类器和 DeepJSCC baseline 已完成。下一项为
fragility 排序的 held-out 验证。

# 当前摘要

- 项目骨架、AWGN 信道、DeepJSCC、CIFAR ResNet-18、feature grouping、
  oracle fragility、三类比较排序和 held-out Top-K 评估已实现。
- DeepJSCC baseline 已实现训练结束后的多 SNR 完整测试，记录 PSNR、
  四尺度 CIFAR MS-SSIM、LPIPS、实值 CBR 和四类语义指标。
- `EXP-S2-002` 排序评估代码已补齐逐样本 Spearman/Kendall、完整 deletion
  curve、deletion AUC、bootstrap CI 和 fragility 相对 baseline 的配对优势；
  正式实验尚未运行。
- `EXP-S0-001` 已在 RTX 4090 上完成独立 GPU dry-run，验证分类器与
  DeepJSCC 反向传播、实验日志、manifest、checkpoint 和真实 LPIPS 流程。
- `EXP-S1-004` 已完成 CIFAR-10 ResNet-18 baseline，最佳 test accuracy
  为 `95.29%`；checkpoint 独立重载复算结果一致。
- `EXP-S1-005` 已完成 CIFAR-10 AWGN DeepJSCC baseline；实值 CBR 为
  `1/3`，0/5/10/15/20 dB 的 PSNR 为
  `22.27/26.23/29.34/31.41/32.42` dB，重建分类准确率为
  `62.79%/82.08%/88.40%/90.85%/91.33%`。
- `EXP-S1-005` 的汇报素材已生成，包括质量/语义曲线和真实传输重建图，
  位于 `outputs/EXP-S1-005/report_assets/`。
- 实验追踪支持唯一 `EXP-Sx-NNN` ID、独立配置、日志、manifest、指标与
  checkpoint 目录，并拒绝覆盖已有实验。
- CIFAR-10 本地 train/test split 可读取：50,000/10,000 张。
- GitHub：`https://github.com/daiqizai/semantic_fragility_jscc`
- 当前完成的正式实验数：2；失败正式实验数：1；工程 dry-run 数：1。
  dry-run 和失败实验均不作为论文结果。

# 任务状态

| 任务 | 状态 | 结果摘要或阻塞原因 | 详情路径 |
|---|---|---|---|
| 项目与实验追踪骨架 | DONE | 当前 8 个单元测试通过，配置解析与覆盖保护正常 | `src/fragile_jscc/`、`tests/` |
| 文档职责与 AI 协作规范 | DONE | 建立五类根文档，进度与完整实验记录分离 | `AGENTS.md`、`PROJECT.md` |
| EXP-S0-001 GPU dry-run | DONE | RTX 4090 上训练、追踪、checkpoint 和 LPIPS 流程通过 | `outputs/EXP-S0-001/summary.json` |
| EXP-S1-001 分类器 baseline | INVALID | 沙箱内 CUDA 不可见，训练开始前失败；产物保留 | `outputs/EXP-S1-001/` |
| EXP-S1-002 旧 DeepJSCC 计划 | INVALID | 未运行；评估协议不完整，由 EXP-S1-003 替代 | `EXPERIMENTS.md` |
| EXP-S1-003 旧 DeepJSCC 计划 | INVALID | 未运行；分类器依赖失效，由 EXP-S1-005 替代 | `EXPERIMENTS.md` |
| EXP-S1-004 分类器 baseline | DONE | 最佳 test accuracy 95.29%，独立 checkpoint 复算一致 | `outputs/EXP-S1-004/summary.json` |
| EXP-S1-005 DeepJSCC baseline | DONE | 100 epoch 与 5 个测试 SNR 完成；CBR 1/3，完整质量和语义指标已记录 | `outputs/EXP-S1-005/summary.json` |
| EXP-S2-001 旧 fragility 排序计划 | INVALID | 未运行；checkpoint 依赖失效，由 EXP-S2-002 替代 | `EXPERIMENTS.md` |
| EXP-S2-002 fragility 排序 | TODO | 两个阶段1 checkpoint 已就绪，可启动 held-out 排序验证 | `EXPERIMENTS.md` |

# 已验证结论

- 代码级 smoke 流程和实验追踪机制可运行。
- CIFAR-10 ResNet-18 baseline 在 `EXP-S1-004` 达到 `95.29%` test
  accuracy，checkpoint 可独立重载并复现该结果。
- CIFAR-10 AWGN DeepJSCC baseline 在固定实值 CBR `1/3` 下完成
  0–20 dB 多 SNR 测试，图像质量和语义指标随 SNR 整体改善。
- 尚无研究假设得到正式实验支持。

# 当前风险

- CIFAR-10 的 MS-SSIM 是明确记录的四尺度版本，不能与未说明尺度设置的结果
  直接比较。
- `lpips==0.1.4` 在当前 torchvision 上会产生 deprecated API warning，
  但 CPU 与 GPU 实测均可正常计算。
- 正式实验必须从干净 Git 工作区启动；GPU 训练还必须在沙箱外运行，
  否则 PyTorch 无法访问 CUDA。
- 训练、训练型 dry-run 和长时间实验只能使用物理 GPU 4–7；物理 GPU 0–3
  已保留，禁止占用。
- 当前 `activation_saliency` 是 latent magnitude，尚不是严格 attention baseline。
- Gradient baseline、局部干预强度和 Monte Carlo 方差仍需公平性验证。
- `EXP-S2-002` 已具备置信区间、Kendall tau 和 deletion AUC 输出，但正式
  排序实验尚未运行，尚无统计结论。

# 下一步

1. 运行 `EXP-S2-002`，优先比较 fragility 与 channel-aware gradient。
2. 核验 held-out 排序评估的运行规模、显存和统计输出是否满足阶段2协议。

# 最近更新

## 2026-06-25：生成 EXP-S1-005 汇报图表和真实传输样例

- 完成内容：基于已完成的 `EXP-S1-005` baseline 汇总指标生成可汇报素材，
  包括 SNR-质量曲线、SNR-语义鲁棒性曲线、真实 DeepJSCC 传输重建网格、
  单张原图/重建图、CSV 指标表和简短汇报要点。
- 修改文件：`scripts/make_report_assets.py`、`README.md`、`PROGRESS.md`、
  `EXPERIMENTS.md`；生成产物位于
  `outputs/EXP-S1-005/report_assets/`，该目录由 `.gitignore` 保护。
- 执行命令：读取共享文档；`git status --short`；`git log --oneline -5`；
  `git diff --check`；`python -m compileall -q src scripts tests`；
  `python -m unittest discover -s tests -v`；
  `scripts/make_report_assets.py --device cpu`；`git check-ignore -v`
  检查报告图片忽略规则。
- 验证结果：编译通过；10 个单元测试通过；报告素材脚本在 CPU 上完成，
  生成 `quality_vs_snr.png`、`semantic_vs_snr.png`、
  `actual_transmission_grid.png`、`report_brief.md`、两份 CSV 和
  `transmission_samples/`。
- 新问题：这些图只说明 DeepJSCC baseline 随 SNR 变化的质量和语义退化，
  不能作为 semantic fragility 排序优于 baseline 的证据；该结论仍需
  `EXP-S2-002`。
- 下一步：将这些素材用于阶段1汇报，同时继续运行 `EXP-S2-002` 排序验证。

## 2026-06-25：准备服务器迁移并补齐阶段2评估输出

- 完成内容：新增服务器迁移清单；补齐 `EXP-S2-002` 排序评估的逐样本
  Spearman/Kendall、完整 deletion curve、deletion AUC、95% bootstrap CI
  和 semantic fragility 相对各 baseline 的配对优势输出。
- 修改文件：`SERVER_MIGRATION.md`、`README.md`、`EXPERIMENTS.md`、
  `configs/EXP-S2-002_ranking.json`、`scripts/run_ranking.py`、
  `src/fragile_jscc/config.py`、`src/fragile_jscc/evaluation.py`、
  `tests/test_core.py`、`PROGRESS.md`。
- 执行命令：读取项目共享文档；`git status --short --branch`；
  `git remote -v`；实验产物大小检查；`git diff --check`；
  `python -m compileall -q src scripts tests`；10 个单元测试；
  `scripts/smoke_test.py`；真实 checkpoint 的 1 样本 CPU ranking 端到端检查。
- 验证结果：编译通过；10 个单元测试通过；smoke test 通过；1 样本 ranking
  端到端检查生成 `ranking_results.json`、`ranking_per_sample.pt` 和 150 条
  汇总记录；未启动正式 `EXP-S2-002`。
- 新问题：换服务器时 Git 不会携带数据、checkpoint 和 outputs，必须按
  `SERVER_MIGRATION.md` 单独复制；当前沙箱内 `nvidia-smi` 不可用，GPU 检查
  需在新服务器或沙箱外执行。
- 下一步：在新服务器完成产物迁移和快速验收后，从干净工作区运行
  `EXP-S2-002`。

## 2026-06-13：完成 EXP-S1-005 DeepJSCC baseline

- 完成内容：在物理 GPU 7 上完成 100 epoch CIFAR-10 AWGN DeepJSCC
  训练，并在 0、5、10、15、20 dB 对完整 10,000 张 test split 计算质量和
  语义指标。
- 修改文件：`PROGRESS.md`、`EXPERIMENTS.md`；实验产物位于
  `outputs/EXP-S1-005/` 和 `checkpoints/EXP-S1-005/`。
- 执行命令：GPU 占用检查；沙箱外
  `CUDA_VISIBLE_DEVICES=7 ... train_jscc.py ...`；manifest、summary、
  metrics 和忽略规则检查；checkpoint CPU 重载；8 个单元测试；
  `git diff --check`。
- 验证结果：manifest 为 `completed`，Git commit `40d89a9`、启动时工作区
  干净；100 条训练和 5 条测试记录完整；最终 train MSE `0.00170859`；
  checkpoint epoch 100 可重载且无 missing/unexpected key；8 个单元测试
  通过；完整结果见 `outputs/EXP-S1-005/summary.json`。
- 新问题：0 dB 重建分类准确率为 `62.79%`、semantic failure rate 为
  `33.78%`，低 SNR 语义退化明显，适合作为阶段2排序验证的重点条件。
- 下一步：运行 `EXP-S2-002` fragility 排序 held-out 验证。

## 2026-06-12：完成 EXP-S1-004 分类器 baseline

- 完成内容：在物理 GPU 7 上完成 100 epoch CIFAR-10 ResNet-18 训练；
  保留全程最佳 checkpoint，并在 CPU 上独立重载后复算完整 test split。
- 修改文件：`PROGRESS.md`、`EXPERIMENTS.md`；实验产物位于
  `outputs/EXP-S1-004/` 和 `checkpoints/EXP-S1-004/`。
- 执行命令：沙箱外 `CUDA_VISIBLE_DEVICES=7 ... train_classifier.py ...`；
  summary/manifest/metrics 检查；checkpoint CPU 独立重载与 10,000 张测试集
  复算；`git diff --check`。
- 验证结果：manifest 为 `completed`，Git commit `7ab8259`、启动时工作区
  干净；最佳 epoch 95，test accuracy `95.29%`；独立复算为
  `9529/10000`，无 missing/unexpected state key。
- 新问题：沙箱内多 worker 的独立复算受 socket 权限限制，改用
  `num_workers=0` 后通过；正式 GPU 实验仍需在沙箱外运行。
- 下一步：运行 `EXP-S1-005` DeepJSCC baseline。

## 2026-06-12：EXP-S1-001 启动失败并分配重跑 ID

- 完成内容：提交 DeepJSCC 指标与 GPU dry-run 改动；从干净工作区启动
  `EXP-S1-001`，但沙箱内 CUDA 不可见，训练 step 开始前失败；保留失败产物，
  创建 `EXP-S1-004` 重跑配置，并为受依赖路径影响的后续计划创建
  `EXP-S1-005` 和 `EXP-S2-002`。
- 修改文件：新增三个配置；更新三个脚本默认配置及共享文档。
- 执行命令：编译、8 个单元测试、smoke test、配置 ID 检查；
  `git commit`；`CUDA_VISIBLE_DEVICES=7 ... train_classifier.py ...`；
  沙箱外 CUDA 可用性检查。
- 验证结果：代码验证通过；`EXP-S1-001` manifest 为 `failed`，Git commit
  `6822ca1`、工作区干净、错误为 `No CUDA GPUs are available`；沙箱外同一
  环境可识别物理 GPU 7 的 RTX 4090。
- 新问题：GPU 正式实验必须在沙箱外启动。
- 下一步：提交新实验 ID 与依赖更新后，在沙箱外运行 `EXP-S1-004`。

## 2026-06-12：记录 GPU 使用约束

- 完成内容：规定训练、训练型 dry-run 和长时间实验禁止使用物理 GPU 0–3，
  仅可通过 `CUDA_VISIBLE_DEVICES` 选择物理 GPU 4–7；解释可见设备内
  `cuda:0` 与物理编号的映射。
- 修改文件：`AGENTS.md`、`README.md`、`PROGRESS.md`、`EXPERIMENTS.md`。
- 执行命令：`git status --short --branch`；GPU 命令与文档全文检索；
  `git diff --check`。
- 验证结果：现有正式实验命令和 `EXP-S0-001` 均使用物理 GPU 7，符合约束；
  未启动训练或新实验。
- 新问题：无。
- 下一步：审查并提交当前改动后，在物理 GPU 4–7 中选择空闲设备运行
  `EXP-S1-001`。

## 2026-06-12：完成独立 GPU dry-run

- 完成内容：新增 `EXP-S0-001` 专用 GPU dry-run；在单批 CIFAR-10 上完成
  ResNet-18 和 ConvDeepJSCC 优化步骤，生成并重载两份 checkpoint，并在 GPU
  上执行真实 AlexNet LPIPS 与完整重建指标。
- 修改文件：`scripts/gpu_dry_run.py`、
  `configs/EXP-S0-001_gpu_dryrun.json`、`README.md`、`PROGRESS.md`、
  `EXPERIMENTS.md`。
- 执行命令：`python -m compileall -q src scripts tests`；
  `python -m unittest discover -s tests -v`；配置 ID 唯一性检查；
  `CUDA_VISIBLE_DEVICES=7 ... scripts/gpu_dry_run.py ...`；
  checkpoint CPU 重载检查；`git check-ignore -v`；`git diff --check`。
- 验证结果：8 个单元测试通过；`EXP-S0-001` 在 RTX 4090 上完成，manifest
  状态为 `completed`；分类器与 DeepJSCC checkpoint 可加载；LPIPS 为
  `0.516387`，峰值 CUDA 已分配内存约 173 MiB。该数值仅为工程验证，
  不作为研究结果。
- 新问题：LPIPS 仍产生 torchvision deprecated API warning；正式实验要求
  干净 Git 工作区，当前改动需先审查并提交。
- 下一步：恢复干净工作区后运行 `EXP-S1-001`。

## 2026-06-12：补齐 DeepJSCC baseline 测试指标

- 完成内容：新增 PSNR、四尺度 CIFAR MS-SSIM、LPIPS 和实值 CBR；
  DeepJSCC 训练结束后按固定测试 SNR 输出图像质量、clean/reconstruction
  accuracy、prediction consistency、semantic failure rate 和 semantic KL；
  新增 `EXP-S1-003` 配置，旧 `EXP-S1-002` 未运行并标为 `INVALID`。
- 修改文件：`src/fragile_jscc/quality.py`、`src/fragile_jscc/evaluation.py`、
  `scripts/train_jscc.py`、`tests/test_core.py`、`configs/EXP-S1-003_deepjscc.json`、
  `configs/EXP-S2-001_ranking.json`、依赖文件及共享文档。
- 执行命令：`python -m compileall -q src scripts tests`；
  `python -m unittest discover -s tests -v`；`scripts/smoke_test.py`；
  配置 JSON 唯一性检查；真实 LPIPS CPU 批次；
  单批 DeepJSCC CPU 端到端评估；MS-SSIM 与 `pytorch-msssim` 底层公式交叉检查；
  `git diff --check`。
- 验证结果：8 个单元测试通过；CPU 端到端评估输出全部预期字段且 CBR 为
  `1/3`；MS-SSIM 交叉检查最大绝对差约 `7e-7`；未运行训练或正式实验。
- 新问题：GPU 仍不可用，尚未验证 CUDA 训练与评估；LPIPS 依赖产生
  torchvision deprecated API warning。
- 下一步：创建独立 GPU dry-run 实验，再运行 `EXP-S1-001`。

## 2026-06-12：扩充 Git 忽略保护

- 完成内容：按 Python 缓存、构建产物、本地环境、IDE、数据、权重和实验产物
  分类扩充 `.gitignore`；禁止 AI 使用 `git add -f` 绕过保护。
- 修改文件：`.gitignore`、`AGENTS.md`、`README.md`、`PROGRESS.md`。
- 执行命令：`git check-ignore -v` 代表路径检查；已跟踪文件审计；
  `git ls-files -ci --exclude-standard`；`git diff --check`；
  `python -m compileall -q src scripts tests`；
  `python -m unittest discover -s tests -v`。
- 验证结果：数据、权重、日志、IDE、密钥和缓存代表路径均命中预期规则；
  没有已跟踪文件被误忽略；最大已跟踪文件约 5.8 KB；6 个单元测试通过；
  未运行训练或正式实验。
- 新问题：无。
- 下一步：完善 DeepJSCC baseline 的完整测试指标。

## 2026-06-12：文档职责重构

- 完成内容：新增 `AGENTS.md`、`PROJECT.md`；精简本文件；扩充
  `EXPERIMENTS.md`；更新 `README.md` 文档入口。
- 修改文件：`AGENTS.md`、`PROJECT.md`、`PROGRESS.md`、
  `EXPERIMENTS.md`、`README.md`。
- 执行命令：`python -m compileall -q src scripts tests`；
  `python -m unittest discover -s tests -v`；Markdown 本地链接检查；
  `git diff --check`。
- 验证结果：6 个单元测试通过；5 份 Markdown 文档无缺失本地链接；
  未运行训练或正式实验。
- 新问题：无新增研究问题；明确 `STALE` 是实验有效性标记，不是任务状态。
- 下一步：完善 DeepJSCC baseline 的完整测试指标。

更早的实验与代码历史通过 Git 追溯；完整实验记录只写入
`EXPERIMENTS.md` 和 `outputs/EXP-xxx/`。
