# Channel-Conditioned Semantic Fragility for DeepJSCC

This repository starts the first validation stage for:

> Channel-Conditioned Semantic Fragility-Aware DeepJSCC under a Fixed
> Transmission Budget

The immediate question is deliberately narrow: does an interventional,
channel-matched fragility score identify latent groups whose corruption causes
semantic failure better than random, activation saliency, and gradient
importance?

## Current Scope

- CIFAR-10/CIFAR-100 input at 32 x 32.
- Convolutional DeepJSCC with an explicit channel-latent interface.
- AWGN with per-sample power calibration.
- Channel-group and spatial-token interventions.
- Oracle semantic fragility based on `KL(reference || perturbed)`.
- Random, activation-saliency, channel-aware gradient-times-activation, and
  fragility ranks.
- Held-out Top-K corruption evaluation using fresh channel noise.
- Accuracy drop, consistency, semantic failure rate, and semantic KL.
- Spearman correlation against independently sampled singleton effects.

The oracle intervention replaces the selected group's AWGN realization with an
independent realization at the same SNR. All non-selected groups retain the
paired baseline realization. Score generation and ranking evaluation call the
channel independently to avoid reusing the same perturbation. Competing ranking
methods share the same held-out noise pair within each batch.

## Layout

```text
README.md                installation and running instructions
PROGRESS.md              authoritative current progress and conclusions
EXPERIMENTS.md           complete experiment index
configs/EXP-Sx-NNN_*.json
outputs/EXP-Sx-NNN/      config snapshot, log, manifest, metrics, results
checkpoints/EXP-Sx-NNN/  model weights
scripts/train_classifier.py
scripts/train_jscc.py
scripts/run_ranking.py   first-stage ranking experiment
scripts/smoke_test.py    no data or checkpoint required
src/fragile_jscc/        channels, models, grouping, scores, evaluation
tests/                   fast core tests
```

## Quick Check

The existing `semantic` environment already contains PyTorch and torchvision:

```bash
cd /data2/liulu/semantic_fragility_jscc
/data2/liulu/miniconda3/envs/semantic/bin/python scripts/smoke_test.py
/data2/liulu/miniconda3/envs/semantic/bin/python -m unittest discover -s tests
```

## First Real Run

The CIFAR-10 files already exist under
`/data2/liulu/semantic_comm/data`, which is the default data root.

```bash
cd /data2/liulu/semantic_fragility_jscc

CUDA_VISIBLE_DEVICES=7 \
/data2/liulu/miniconda3/envs/semantic/bin/python \
  scripts/train_classifier.py \
  --config configs/EXP-S1-001_classifier.json --device cuda:0

CUDA_VISIBLE_DEVICES=7 \
/data2/liulu/miniconda3/envs/semantic/bin/python \
  scripts/train_jscc.py \
  --config configs/EXP-S1-002_deepjscc.json --device cuda:0

CUDA_VISIBLE_DEVICES=7 \
/data2/liulu/miniconda3/envs/semantic/bin/python \
  scripts/run_ranking.py \
  --config configs/EXP-S2-001_ranking.json --device cuda:0
```

Each config contains a unique experiment ID. The runner refuses to overwrite
an existing `outputs/EXP-xxx/` or `checkpoints/EXP-xxx/` directory. A retry,
including a retry after failure, must receive a new ID and config. Results for
the ranking run are written to
`outputs/EXP-S2-001/ranking_results.json`.

Before starting or changing work, read `PROGRESS.md`. After every code change,
training run, experiment, or important decision, update it before ending the
task.

## Experimental Guardrails

1. Keep the classifier and DeepJSCC frozen during ranking experiments.
2. Generate oracle labels and evaluate Top-K deletion with independent noise.
3. Compare against the strongest channel-aware gradient baseline before
   investing in a predictor or resource allocator.
4. Continue to fixed-budget protection only if fragility consistently improves
   held-out deletion metrics across multiple SNR values.

## Next Milestone

After the ranking hypothesis survives, add:

- a lightweight `P_frag(z, SNR)` predictor;
- oracle/predicted Spearman and Kendall correlations;
- fixed-total-power allocation first;
- explicit side-information accounting;
- fixed-symbol-budget allocation and Rayleigh fading second.
