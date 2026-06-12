# 当前阶段

**阶段1：Baseline**

状态：`IN_PROGRESS`

阶段0准备与定义已完成。阶段1代码入口和计划配置已准备，但尚无正式
baseline checkpoint 或正式实验结果。

# 当前摘要

- 项目骨架、AWGN 信道、DeepJSCC、CIFAR ResNet-18、feature grouping、
  oracle fragility、三类比较排序和 held-out Top-K 评估已实现。
- DeepJSCC baseline 已实现训练结束后的多 SNR 完整测试，记录 PSNR、
  四尺度 CIFAR MS-SSIM、LPIPS、实值 CBR 和四类语义指标。
- `EXP-S0-001` 已在 RTX 4090 上完成独立 GPU dry-run，验证分类器与
  DeepJSCC 反向传播、实验日志、manifest、checkpoint 和真实 LPIPS 流程。
- 实验追踪支持唯一 `EXP-Sx-NNN` ID、独立配置、日志、manifest、指标与
  checkpoint 目录，并拒绝覆盖已有实验。
- CIFAR-10 本地 train/test split 可读取：50,000/10,000 张。
- GitHub：`https://github.com/daiqizai/semantic_fragility_jscc`
- 当前正式实验数：0；工程 dry-run 数：1。二者均不作为论文实验结果。

# 任务状态

| 任务 | 状态 | 结果摘要或阻塞原因 | 详情路径 |
|---|---|---|---|
| 项目与实验追踪骨架 | DONE | 当前 8 个单元测试通过，配置解析与覆盖保护正常 | `src/fragile_jscc/`、`tests/` |
| 文档职责与 AI 协作规范 | DONE | 建立五类根文档，进度与完整实验记录分离 | `AGENTS.md`、`PROJECT.md` |
| EXP-S0-001 GPU dry-run | DONE | RTX 4090 上训练、追踪、checkpoint 和 LPIPS 流程通过 | `outputs/EXP-S0-001/summary.json` |
| EXP-S1-001 分类器 baseline | TODO | 尚未启动 | `EXPERIMENTS.md` |
| EXP-S1-002 旧 DeepJSCC 计划 | INVALID | 未运行；评估协议不完整，由 EXP-S1-003 替代 | `EXPERIMENTS.md` |
| EXP-S1-003 DeepJSCC baseline | BLOCKED | 完整 test 指标已实现；等待 EXP-S1-001 checkpoint | `EXPERIMENTS.md` |
| EXP-S2-001 fragility 排序 | BLOCKED | 等待 EXP-S1-001 和 EXP-S1-003 checkpoint | `EXPERIMENTS.md` |

# 已验证结论

- 代码级 smoke 流程和实验追踪机制可运行。
- 尚无研究假设得到正式实验支持。

# 当前风险

- CIFAR-10 的 MS-SSIM 是明确记录的四尺度版本，不能与未说明尺度设置的结果
  直接比较。
- `lpips==0.1.4` 在当前 torchvision 上会产生 deprecated API warning，
  但 CPU 与 GPU 实测均可正常计算。
- 正式实验要求 Git 工作区干净；当前指标实现和 dry-run 相关改动尚未提交，
  运行 `EXP-S1-001` 前必须先完成审查与提交。
- 训练、训练型 dry-run 和长时间实验只能使用物理 GPU 4–7；物理 GPU 0–3
  已保留，禁止占用。
- 当前 `activation_saliency` 是 latent magnitude，尚不是严格 attention baseline。
- Gradient baseline、局部干预强度和 Monte Carlo 方差仍需公平性验证。
- 尚无置信区间、Kendall tau、deletion AUC 或统计显著性检验。

# 下一步

1. 审查并提交当前 DeepJSCC 指标与 GPU dry-run 改动，恢复干净工作区。
2. 运行 `EXP-S1-001` 分类器 baseline。
3. 运行 `EXP-S1-003` DeepJSCC baseline。
4. baseline 固定后运行阶段2 pilot，优先比较 fragility 与 channel-aware gradient。

# 最近更新

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
