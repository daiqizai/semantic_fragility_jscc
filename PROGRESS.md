# 当前阶段

**阶段0：准备与定义**

状态：`IN_PROGRESS`

阶段1代码入口和计划配置已准备，但尚无正式 baseline checkpoint 或正式实验结果。

# 当前摘要

- 项目骨架、AWGN 信道、DeepJSCC、CIFAR ResNet-18、feature grouping、
  oracle fragility、三类比较排序和 held-out Top-K 评估已实现。
- 实验追踪支持唯一 `EXP-Sx-NNN` ID、独立配置、日志、manifest、指标与
  checkpoint 目录，并拒绝覆盖已有实验。
- CIFAR-10 本地 train/test split 可读取：50,000/10,000 张。
- GitHub：`https://github.com/daiqizai/semantic_fragility_jscc`
- 当前正式实验数：0。smoke test 不作为论文实验结果。

# 任务状态

| 任务 | 状态 | 结果摘要或阻塞原因 | 详情路径 |
|---|---|---|---|
| 项目与实验追踪骨架 | DONE | 6 个单元测试通过，配置解析与覆盖保护正常 | `src/fragile_jscc/`、`tests/` |
| 文档职责与 AI 协作规范 | DONE | 建立五类根文档，进度与完整实验记录分离 | `AGENTS.md`、`PROJECT.md` |
| EXP-S1-001 分类器 baseline | TODO | 尚未启动 | `EXPERIMENTS.md` |
| EXP-S1-002 DeepJSCC baseline | TODO | 启动前需补完整 test 指标 | `EXPERIMENTS.md` |
| EXP-S2-001 fragility 排序 | BLOCKED | 等待两个阶段1 checkpoint | `EXPERIMENTS.md` |

# 已验证结论

- 代码级 smoke 流程和实验追踪机制可运行。
- 尚无研究假设得到正式实验支持。

# 当前风险

- 当前会话中的 `semantic` 环境曾报告 CUDA 不可用，GPU 训练入口尚未 dry run。
- DeepJSCC baseline 尚缺 test PSNR、MS-SSIM、LPIPS、CBR 和语义指标。
- 当前 `activation_saliency` 是 latent magnitude，尚不是严格 attention baseline。
- Gradient baseline、局部干预强度和 Monte Carlo 方差仍需公平性验证。
- 尚无置信区间、Kendall tau、deletion AUC 或统计显著性检验。

# 下一步

1. 为 DeepJSCC baseline 增加 test PSNR、MS-SSIM、LPIPS、CBR 和分类指标。
2. 创建独立 GPU dry-run 实验 ID，验证环境、日志和 checkpoint 流程。
3. 运行 `EXP-S1-001` 分类器 baseline。
4. 为完善后的 DeepJSCC baseline 创建新实验 ID 并训练。
5. baseline 固定后运行阶段2 pilot，优先比较 fragility 与 channel-aware gradient。

# 最近更新

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
