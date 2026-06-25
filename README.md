# Channel-Conditioned Semantic Fragility for DeepJSCC

This repository starts the first validation stage for:

> Channel-Conditioned Semantic Fragility-Aware DeepJSCC under a Fixed
> Transmission Budget

Research definitions and fixed experimental conventions live in
[`PROJECT.md`](PROJECT.md). AI collaborators must read
[`AGENTS.md`](AGENTS.md) before making changes.

## Layout

```text
AGENTS.md                mandatory AI collaboration rules
PROJECT.md               research definition and fixed conventions
README.md                installation and running instructions
PROGRESS.md              concise current progress and next actions
EXPERIMENTS.md           complete experiment index
SERVER_MIGRATION.md      checklist for moving this project to another server
configs/EXP-Sx-NNN_*.json
outputs/EXP-Sx-NNN/      config snapshot, log, manifest, metrics, results
checkpoints/EXP-Sx-NNN/  model weights
scripts/train_classifier.py
scripts/train_jscc.py
scripts/run_ranking.py   first-stage ranking experiment
scripts/make_report_assets.py
                         EXP-S1-005 report plots and transmission samples
scripts/smoke_test.py    no data or checkpoint required
scripts/gpu_dry_run.py   one-batch CUDA, checkpoint and LPIPS validation
src/fragile_jscc/        channels, models, grouping, scores, evaluation
tests/                   fast core tests
```

Read project memory in this order before starting work:

1. `AGENTS.md`
2. `PROJECT.md`
3. `PROGRESS.md`
4. `EXPERIMENTS.md`
5. `README.md`

## Quick Check

The existing `semantic` environment already contains PyTorch and torchvision:

```bash
cd /data2/liulu/semantic_fragility_jscc
/data2/liulu/miniconda3/envs/semantic/bin/python scripts/smoke_test.py
/data2/liulu/miniconda3/envs/semantic/bin/python -m unittest discover -s tests
```

## Installation

The project currently uses the existing `semantic` Conda environment:

```bash
cd /data2/liulu/semantic_fragility_jscc
/data2/liulu/miniconda3/envs/semantic/bin/python -m pip install -e .
```

Do not install or change dependencies during a running experiment.

Core requirements are also listed in `requirements.txt` and `pyproject.toml`.

## Running

The CIFAR-10 files already exist under
`/data2/liulu/semantic_comm/data`, which is the default data root.

Training and training-style dry-runs must use physical GPU 4, 5, 6, or 7.
Physical GPU 0-3 are reserved and must not be used. Always set
`CUDA_VISIBLE_DEVICES` explicitly; after selecting one physical GPU,
`--device cuda:0` refers to that selected device.

```bash
cd /data2/liulu/semantic_fragility_jscc

CUDA_VISIBLE_DEVICES=7 \
/data2/liulu/miniconda3/envs/semantic/bin/python \
  scripts/train_classifier.py \
  --config configs/EXP-S1-004_classifier.json --device cuda:0

CUDA_VISIBLE_DEVICES=7 \
/data2/liulu/miniconda3/envs/semantic/bin/python \
  scripts/train_jscc.py \
  --config configs/EXP-S1-005_deepjscc.json --device cuda:0

CUDA_VISIBLE_DEVICES=7 \
/data2/liulu/miniconda3/envs/semantic/bin/python \
  scripts/run_ranking.py \
  --config configs/EXP-S2-002_ranking.json --device cuda:0
```

Each config contains a unique experiment ID. The runner refuses to overwrite
an existing `outputs/EXP-xxx/` or `checkpoints/EXP-xxx/` directory. A retry,
including a retry after failure, must receive a new ID and config. Results for
the ranking run are written to `outputs/EXP-S2-002/ranking_results.json`;
per-sample correlations, deletion curves and AUC values are stored in
`outputs/EXP-S2-002/ranking_per_sample.pt`. The summary includes Spearman and
Kendall correlations to independent held-out singleton effects, complete
group-deletion curves, deletion AUC, requested Top-K points, 95% bootstrap
confidence intervals and paired semantic-fragility advantages over each
baseline.

The complete DeepJSCC test pass requires the classifier checkpoint from
`EXP-S1-004`. It evaluates the final model at every configured test SNR and
writes PSNR, four-scale CIFAR MS-SSIM, LPIPS, CBR, accuracy, prediction
consistency, semantic failure rate and semantic KL to `metrics.jsonl` and
`summary.json`. LPIPS uses pretrained AlexNet weights and may download them on
the first run if they are not already cached.

To regenerate the current stage-1 report figures from the completed
`EXP-S1-005` artifacts:

```bash
/data2/liulu/miniconda3/envs/semantic/bin/python \
  scripts/make_report_assets.py \
  --device cpu
```

This writes quality curves, semantic robustness curves, real transmission
examples, CSV tables and a short brief to
`outputs/EXP-S1-005/report_assets/`. These files are ignored by Git and should
be copied separately if the project is moved to another server.

Datasets, checkpoints, logs, tracker directories and generated experiment
artifacts are intentionally ignored by Git. Do not force-add them; share their
paths and summary metrics through `EXPERIMENTS.md`.

Commands and full metadata for every planned or completed formal experiment
are indexed in `EXPERIMENTS.md`. Do not reuse an experiment ID or overwrite an
existing artifact directory.

## Documentation Policy

- Keep `PROGRESS.md` short: summaries and paths only.
- Put experiment commands, metadata and result tables in `EXPERIMENTS.md`.
- Put full logs and per-epoch metrics in `outputs/EXP-xxx/`.
- Update `PROJECT.md` before changing a public interface or research protocol.
