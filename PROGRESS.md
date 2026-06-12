# 项目目标

研究 Channel-Conditioned Semantic Fragility-Aware DeepJSCC：判断 DeepJSCC
传输 latent 中哪些 feature group 在与当前 SNR/CSI 匹配的信道扰动下最容易
导致接收端语义任务失败，并在固定 CBR、总功率或总信道符号预算内给予这些
特征更强保护。

核心假设是：任务相关性不等于信道扰动下的语义失败风险；基于干预得到的
channel-conditioned semantic fragility，比 attention/saliency 和 gradient
importance 更准确地预测特征受损造成的分类失败，并可指导固定预算不等保护。

最终目标是完成可复现的 CIFAR/ImageNet 子集、AWGN/Rayleigh、多 SNR、多 CBR
实验，包含资源分配优化、边信息开销、复杂度和完整消融，形成一篇实验与系统
驱动的 C 类会议论文；若方法增益稳定且理论与实验完整，则尝试普通 B 类会议。

# 当前阶段

**阶段0 准备与定义。**

阶段1 Baseline 的代码和计划配置已准备，但分类器与 DeepJSCC baseline 尚未正式
训练，因此不能标记为阶段1完成或阶段2开始。

# 已完成

- [x] **2026-06-12：建立独立项目骨架**
  - 代码路径：`src/fragile_jscc/`、`scripts/`、`tests/`
  - 配置文件：无正式实验配置；仅代码级 smoke 输入
  - 输出结果路径：无持久化实验结果
  - 验证方式：`python -m compileall -q src scripts tests`；
    `python scripts/smoke_test.py`
  - 稳定复现：是，CPU 环境可重复执行；smoke 数值不作为论文实验结果

- [x] **2026-06-12：实现第一阶段 fragility 排序验证骨架**
  - 代码路径：`src/fragile_jscc/channels.py`、
    `src/fragile_jscc/scoring.py`、`src/fragile_jscc/evaluation.py`、
    `src/fragile_jscc/groups.py`
  - 配置文件：`configs/EXP-S2-001_ranking.json`
  - 输出结果路径：计划为 `outputs/EXP-S2-001/`，尚未运行
  - 验证方式：oracle fragility、saliency、channel-aware
    gradient-times-activation、random 均通过无 checkpoint smoke 流程；
    held-out Top-K 比较共享验证噪声
  - 稳定复现：代码路径可复现；正式实验尚不可复现，因为 baseline
    checkpoint 尚不存在

- [x] **2026-06-12：建立不可覆盖的实验追踪协议**
  - 代码路径：`src/fragile_jscc/experiment.py`、
    `tests/test_experiment.py`
  - 配置文件：`configs/EXP-S1-001_classifier.json`、
    `configs/EXP-S1-002_deepjscc.json`、
    `configs/EXP-S2-001_ranking.json`
  - 输出结果路径：运行时自动创建 `outputs/EXP-xxx/` 和
    `checkpoints/EXP-xxx/`
  - 验证方式：6 个单元测试全部通过，其中实验追踪测试检查 ID 格式、
    manifest、日志/指标路径及拒绝覆盖；三个 JSON 配置均成功解析；
    `git diff --check` 无错误
  - 稳定复现：是

- [x] **2026-06-12：确认 CIFAR-10 本地数据可读取**
  - 代码路径：`src/fragile_jscc/data.py`
  - 配置文件：上述三个 CIFAR-10 配置
  - 输出结果路径：无
  - 验证方式：读取本地 train/test split，样本数分别为 50,000/10,000，
    batch shape 为 `(4, 3, 32, 32)`
  - 稳定复现：是；依赖 `/data2/liulu/semantic_comm/data`

- [x] **2026-06-12：建立 GitHub 远端与首个可追溯代码基线**
  - 代码路径：项目根目录全部已跟踪源码与文档
  - 配置文件：`configs/EXP-S1-001_classifier.json`、
    `configs/EXP-S1-002_deepjscc.json`、
    `configs/EXP-S2-001_ranking.json`
  - 输出结果路径：无正式实验输出
  - 验证方式：远端
    `https://github.com/daiqizai/semantic_fragility_jscc.git` 初始为空；
    本地分支重命名为 `main`；基线 commit 为
    `4c99c662358db00ff27f9273aec0631295ffef6a`；提交后工作区干净；
    6 个单元测试再次通过
  - 稳定复现：是；正式实验 manifest 应引用此 commit 或后续干净 commit

# 正在进行

当前没有训练或实验进程正在运行。

| 任务 | 命令 | 状态 | 日志路径 | Checkpoint | 预计输出 |
|---|---|---|---|---|---|
| EXP-S1-001 分类器 baseline | `CUDA_VISIBLE_DEVICES=7 /data2/liulu/miniconda3/envs/semantic/bin/python scripts/train_classifier.py --config configs/EXP-S1-001_classifier.json --device cuda:0` | 计划，未启动 | `outputs/EXP-S1-001/run.log` | `checkpoints/EXP-S1-001/best.pt` | 每轮 train CE、test accuracy、最佳分类准确率 |
| EXP-S1-002 DeepJSCC baseline | `CUDA_VISIBLE_DEVICES=7 /data2/liulu/miniconda3/envs/semantic/bin/python scripts/train_jscc.py --config configs/EXP-S1-002_deepjscc.json --device cuda:0` | 计划，未启动 | `outputs/EXP-S1-002/run.log` | `checkpoints/EXP-S1-002/latest.pt` | 每轮 train MSE 与最终 checkpoint |
| EXP-S2-001 排序验证 | `CUDA_VISIBLE_DEVICES=7 /data2/liulu/miniconda3/envs/semantic/bin/python scripts/run_ranking.py --config configs/EXP-S2-001_ranking.json --device cuda:0` | 阻塞，等待两个 baseline | `outputs/EXP-S2-001/run.log` | 读取阶段1 checkpoint | 多 SNR、Top-K、四种排序的语义指标 |

# 实验结果

截至 2026-06-12 尚未运行任何正式实验。以下表格不得用 smoke test 或随机未训练
模型的输出填充。

| 实验 ID | Git commit | 数据集与数据划分 | 模型 | 信道类型 | 训练 SNR | 测试 SNR | CBR | 随机种子 | PSNR | MS-SSIM | LPIPS | 分类准确率 | Prediction consistency | Semantic failure rate | 配置、日志和 checkpoint 路径 |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|

# 关键结论

## 已被实验支持的结论

- 暂无。当前只有代码级验证，不能支持研究结论。

## 暂时观察

- 随机未训练模型上的 smoke test 能贯通 score generation 和 Top-K
  evaluation，但其数值没有语义意义。

## 尚未验证的假设

- Semantic fragility 比 random、activation saliency 和最强
  channel-aware gradient baseline 更准确地预测语义失败。
- Fragility 具有稳定的 SNR/CSI conditioning，而非仅由 latent 能量决定。
- Fragility-based allocation 在相同 CBR、功率和符号数下优于其他分配方法。
- 轻量 predictor 能保持足够高的 Spearman/Kendall 排序相关性。

## 已被否定的假设

- 暂无。

# 问题与风险

- GitHub 远端已配置为
  `https://github.com/daiqizai/semantic_fragility_jscc.git`；首个代码基线
  commit 为 `4c99c662358db00ff27f9273aec0631295ffef6a`。本条进度记录会形成后续
  文档 commit，正式实验仍应检查 manifest 中的 commit 与工作区状态。
- 当前执行会话中 `semantic` 环境报告 `torch.cuda.is_available() == False`，
  尚未验证 GPU 训练入口；服务器 GPU 本身可由 `nvidia-smi` 看到。
- 尚无训练完成的 CIFAR-10 ResNet-18 和 DeepJSCC checkpoint。
- 当前 DeepJSCC baseline 只优化 MSE，尚未计算 PSNR、MS-SSIM、LPIPS 和 CBR。
- `activation_saliency` 只是 latent magnitude，不等同于严格 attention baseline；
  后续需要实现与论文可比的 attention/saliency 方法。
- 当前 gradient baseline 使用 clean-image classifier prediction 作为伪标签，需
  检查是否是最强且公平的 channel-aware gradient 定义。
- Oracle fragility 以同 SNR 独立 AWGN realization 替换局部分组噪声；若扰动过弱，
  KL 可能接近浮点噪声，需要研究扰动强度与 Monte Carlo 方差。
- Singleton fragility 与 Top-K 联合破坏之间可能存在特征交互，排序分数未必可加。
- 目前没有 bootstrap 置信区间、Kendall 相关性或统计显著性检验。
- 实验追踪脚本会拒绝复用已有 ID，包括失败实验；排查后必须新建配置和 ID。
- 本次修改发生在正式结果产生前，因此没有旧结果需要因评估代码变化而重跑。

# 下一步

1. 为 DeepJSCC 增加独立 test split 评估及 PSNR、MS-SSIM、LPIPS、CBR 计算，然后
   使用新实验 ID 运行 baseline；不要直接启动当前仅记录 train MSE 的长期训练。
2. 检查 GPU 环境可见性和 `semantic` 环境 CUDA 构建，先执行短 epoch GPU dry run。
3. 运行 `EXP-S1-001`，训练 CIFAR-10 ResNet-18，保存最佳 checkpoint 和逐轮准确率。
4. 在 baseline 固定后运行阶段2小规模 pilot，优先比较 semantic fragility 与
   channel-aware gradient 的 held-out deletion AUC 和排序相关性。
5. 增加 bootstrap 置信区间、Kendall tau、逐样本结果保存和失败样本分析。
6. 仅在阶段2跨多个 SNR 显著优于最强 baseline 后实现 fixed-total-power allocation。

# 变更记录

- **2026-06-12：** 配置 GitHub 远端
  `daiqizai/semantic_fragility_jscc.git`，确认远端为空后建立 `main` 分支首个
  代码基线 commit `4c99c662358db00ff27f9273aec0631295ffef6a`；原因是让后续
  实验能够绑定明确源码版本。
- **2026-06-12：** 完成实验追踪改造后的验证：6 个单元测试通过、所有脚本
  编译通过、三个计划配置解析通过、`git diff --check` 通过；正式实验数量仍为
  0，未生成或填写任何指标数值。
- **2026-06-12：** 创建 `PROGRESS.md` 作为唯一项目进度记录；创建
  `EXPERIMENTS.md` 作为实验索引；原因是建立可追溯、不可覆盖的研究流程。
- **2026-06-12：** 为训练与排序脚本加入唯一实验 ID、独立输出/checkpoint
  目录、日志、manifest、指标 JSONL 和覆盖保护；原因是满足正式实验复现要求。
- **2026-06-12：** 将阶段1与阶段2计划拆分为 `EXP-S1-001`、
  `EXP-S1-002`、`EXP-S2-001`；原因是模型训练与排序评估必须分别追踪。
- **2026-06-12：** 初始化项目骨架、AWGN 干预式 fragility、排序 baseline 和
  held-out Top-K 评估；原因是先验证核心假设，再投入 predictor 与资源分配。
