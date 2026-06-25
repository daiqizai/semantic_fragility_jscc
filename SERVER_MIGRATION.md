# 服务器迁移清单

本文件记录从当前服务器迁移本项目到新服务器时需要带走和验证的内容。
Git 只保存代码、配置和共享文档；数据集、checkpoint、日志和实验输出受
`.gitignore` 保护，必须单独复制。

## 1. 迁移前在当前服务器确认

```bash
cd /data2/liulu/semantic_fragility_jscc
git status --short --branch
git log -3 --oneline
git push
```

确认 `main` 已推到：

```text
git@github.com:daiqizai/semantic_fragility_jscc.git
```

如果工作区仍有未提交改动，先确认这些改动不是其他对话留下的未知修改；
需要保留的改动应提交后再换服务器。

## 2. 需要单独复制的本地产物

必须复制：

```text
/data2/liulu/semantic_comm/data
/data2/liulu/semantic_fragility_jscc/checkpoints/EXP-S1-004/best.pt
/data2/liulu/semantic_fragility_jscc/checkpoints/EXP-S1-005/latest.pt
/data2/liulu/semantic_fragility_jscc/outputs/EXP-S1-004/
/data2/liulu/semantic_fragility_jscc/outputs/EXP-S1-005/
```

这些产物当前大致大小：

```text
checkpoints/EXP-S1-004/  43M
checkpoints/EXP-S1-005/  1.4M
outputs/EXP-S1-004/      3.7M
outputs/EXP-S1-005/      3.8M
```

可选复制：

```text
outputs/EXP-S0-001/
checkpoints/EXP-S0-001/
outputs/EXP-S1-001/
```

可选项只用于追溯 dry-run 或失败实验，不是继续阶段2的硬依赖。

## 3. 复制命令模板

从当前服务器推到新服务器时：

```bash
rsync -aP /data2/liulu/semantic_comm/data/ \
  <new_server>:/data2/liulu/semantic_comm/data/

rsync -aP /data2/liulu/semantic_fragility_jscc/checkpoints/EXP-S1-004/ \
  <new_server>:/data2/liulu/semantic_fragility_jscc/checkpoints/EXP-S1-004/

rsync -aP /data2/liulu/semantic_fragility_jscc/checkpoints/EXP-S1-005/ \
  <new_server>:/data2/liulu/semantic_fragility_jscc/checkpoints/EXP-S1-005/

rsync -aP /data2/liulu/semantic_fragility_jscc/outputs/EXP-S1-004/ \
  <new_server>:/data2/liulu/semantic_fragility_jscc/outputs/EXP-S1-004/

rsync -aP /data2/liulu/semantic_fragility_jscc/outputs/EXP-S1-005/ \
  <new_server>:/data2/liulu/semantic_fragility_jscc/outputs/EXP-S1-005/
```

如果新服务器路径不同，需要同步修改配置里的路径，尤其是：

```text
configs/EXP-S1-004_classifier.json
configs/EXP-S1-005_deepjscc.json
configs/EXP-S2-002_ranking.json
```

若修改数据根目录或 checkpoint 路径，必须创建新的实验 ID，不能复用已经运行过
或已计划绑定旧路径的实验 ID。

## 4. 新服务器初始化

```bash
cd /data2/liulu
git clone git@github.com:daiqizai/semantic_fragility_jscc.git
cd /data2/liulu/semantic_fragility_jscc
```

优先复用已有 `semantic` Conda 环境；若需要新建环境，按 `requirements.txt`
和 `pyproject.toml` 安装，并记录 Python、PyTorch、CUDA 和 GPU 信息。

```bash
/data2/liulu/miniconda3/envs/semantic/bin/python -m pip install -e .
```

不要在正式实验运行中途安装或升级依赖。

## 5. 新服务器验收

先确认数据和 checkpoint：

```bash
test -d /data2/liulu/semantic_comm/data
test -f checkpoints/EXP-S1-004/best.pt
test -f checkpoints/EXP-S1-005/latest.pt
```

运行快速验证：

```bash
/data2/liulu/miniconda3/envs/semantic/bin/python scripts/smoke_test.py
/data2/liulu/miniconda3/envs/semantic/bin/python -m unittest discover -s tests -v
```

检查 GPU，只能使用物理 GPU 4、5、6、7：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free \
  --format=csv,noheader,nounits
```

正式训练、训练型 dry-run 和长时间实验必须显式指定：

```bash
CUDA_VISIBLE_DEVICES=7 ... --device cuda:0
```

其中 `cuda:0` 是可见设备内编号，实际对应物理 GPU 7。

## 6. 迁移后继续当前任务

当前阶段是阶段2，下一项未完成任务是 `EXP-S2-002` fragility 排序验证。
两个阶段1 checkpoint 已就绪后，运行：

```bash
CUDA_VISIBLE_DEVICES=7 \
/data2/liulu/miniconda3/envs/semantic/bin/python \
  scripts/run_ranking.py \
  --config configs/EXP-S2-002_ranking.json \
  --device cuda:0
```

预期输出：

```text
outputs/EXP-S2-002/run.log
outputs/EXP-S2-002/run_manifest.json
outputs/EXP-S2-002/metrics.jsonl
outputs/EXP-S2-002/ranking_results.json
outputs/EXP-S2-002/ranking_per_sample.pt
```

运行结束后必须更新：

```text
PROGRESS.md
EXPERIMENTS.md
```

不要把 `outputs/`、`checkpoints/`、数据集或本地环境提交到 Git。
