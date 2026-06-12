# 项目定义

## 研究目标

本项目研究 **Channel-Conditioned Semantic Fragility-Aware DeepJSCC under
Fixed Transmission Budget**。

传统 attention、saliency 和 gradient importance 主要度量特征与任务的相关性，
但任务相关不等于该特征经信道传输受损后会引发语义失败。本项目的核心问题是：

> 能否估计每个 DeepJSCC 传输特征在给定信道条件下导致语义任务失败的风险，
> 并在固定总预算内据此实施不等保护？

核心假设：

1. Channel-conditioned semantic fragility 比普通任务重要性更准确地预测特征损坏
   引起的分类或语义失败。
2. 在相同 CBR、总功率和总信道符号数下，fragility-based allocation 能在多个
   SNR 和信道模型上稳定优于 random、attention/saliency 和 gradient allocation。

投稿目标是先形成一篇实验与系统驱动的 C 类会议论文。只有在资源分配优化、
多数据集、跨信道泛化、完整消融和复杂度分析均充分时，才考虑普通 B 类会议。

## 方法定义

输入图像 \(x\) 经 DeepJSCC encoder 得到待传输表示：

\[
F = E(x)
\]

对第 \(i\) 个 channel group、spatial token 或 feature group 施加与当前
SNR/CSI 和信道模型匹配的局部干预，得到接收重建 \(\hat{x}'_i\)。以未施加局部
干预的配对接收结果 \(\hat{x}\) 为参照：

\[
r_i(x,\gamma)=
\mathbb{E}_{n,\delta_i}
\left[
D_{\mathrm{sem}}
\left(T(\hat{x}), T(\hat{x}'_i)\right)
\right]
\]

其中：

- \(\gamma\) 表示 SNR/CSI；
- \(T\) 是固定语义模型；
- \(D_{\mathrm{sem}}\) 可为 KL divergence、任务损失变化或 embedding distance；
- \(r_i\) 是 channel-conditioned semantic fragility；
- 基础信道噪声应配对共享，只改变目标 feature group 的局部误差，以降低标签方差；
- 标签生成和 held-out 排序验证必须使用独立随机样本。

方法定位统一表述为 **Channel-Conditioned Interventional Semantic
Fragility**，不把创新主要包装为 counterfactual learning。

## 预定方法链路

1. **Offline Fragility Label Generation**
   - 对不同图像、SNR/CSI 和 feature group 执行局部信道干预；
   - 生成 oracle fragility 排序标签。
2. **Lightweight Fragility Predictor**
   - 学习 \(P_{\mathrm{frag}}(F,\mathrm{SNR/CSI}) \rightarrow r\)；
   - 推理阶段不再逐特征执行干预。
3. **Fixed-Budget Resource Allocation**
   - 在固定总功率、CBR 或信道符号数约束下重分配资源；
   - 所有 token 索引、排序和分配边信息必须计入预算。

## 阶段定义

| 阶段 | 名称 | 完成条件 |
|---|---|---|
| 阶段0 | 准备与定义 | 研究协议、代码骨架、实验追踪与复现规则建立 |
| 阶段1 | Baseline | 分类器和 DeepJSCC baseline 训练、完整质量与语义指标可复现 |
| 阶段2 | Fragility 可行性验证 | Fragility 在独立损坏验证中稳定优于最强排序基线 |
| 阶段3 | 固定预算保护 | 相同预算下 fragility allocation 稳定优于其他分配 |
| 阶段4 | 完整模型 | Predictor、可部署分配器和边信息机制完成 |
| 阶段5 | 完整实验 | 多数据集、信道、SNR、CBR、消融、复杂度和显著性完整 |
| 阶段6 | 论文整理 | 图表、论文、复现说明和投稿材料完成 |

## 最小可行性验证

第一轮固定为：

- CIFAR-10，必要时扩展 CIFAR-100；
- convolutional DeepJSCC；
- ResNet-18 分类器；
- AWGN；
- channel-group 或 spatial-token 粒度；
- KL divergence；
- Random、activation/attention saliency、channel-aware gradient 和 semantic
  fragility 四类排序。

使用独立信道随机样本破坏各方法排名最高的 Top-K 特征，至少报告：

- classification accuracy drop；
- prediction consistency；
- semantic failure rate；
- semantic KL；
- 排序与 held-out singleton effect 的 Spearman/Kendall 相关性；
- deletion curve 或 deletion AUC 及置信区间。

## 固定预算原则

资源分配比较必须同时满足：

- 相同数据划分和模型 checkpoint；
- 相同信道模型与测试随机种子；
- 相同 CBR；
- 相同总发射功率；
- 相同总信道符号数；
- 边信息开销明确计入或单独给出严格上界；
- 不得用不同预算结果宣称方法优越。

优先验证固定总功率分配，再扩展固定符号数和可变压缩率。

## 指标约定

- 图像质量：PSNR、MS-SSIM、LPIPS。
- 语义性能：classification accuracy、prediction consistency、semantic
  failure rate、semantic KL。
- 排序性能：Spearman、Kendall、deletion AUC。
- 系统开销：CBR、总功率、符号数、边信息、参数量、FLOPs、推理时间。

指标定义或评估代码改变后，受影响实验必须在 `EXPERIMENTS.md` 标为
`STALE`，不得继续支撑结论。

## 继续或停止标准

只有同时满足以下条件才进入完整论文阶段：

1. Fragility 在多个 SNR 上比 attention/saliency 和最强 channel-aware gradient
   更准确地预测损坏导致的语义失败，并具有统计显著性。
2. 相同传输预算下，fragility-based allocation 在多个 SNR 上稳定优于其他分配。

若第一项失败，优先检查干预模型、语义距离、粒度和 Monte Carlo 方差。经过预先
记录的有限轮修改仍失败，则将该方向标为 `INVALID`，不得通过选择性报告维持结论。

## 不可随意改变的约定

以下改动属于研究协议变化，必须先更新本文件、说明理由，并为后续实验创建新 ID：

- fragility 的参照分布或干预定义；
- semantic distance 主定义；
- feature grouping；
- 数据划分；
- 预算口径；
- semantic failure rate 定义；
- baseline 公平性条件；
- 继续或停止标准。

