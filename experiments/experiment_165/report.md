x0_lambdas init reduced from 0.1 to 0.05 for less raw embedding noise in deep layers.

## What changed

In `GPT.init_weights` (train.py, line 234):
- Changed `self.x0_lambdas.fill_(0.1)` to `self.x0_lambdas.fill_(0.05)`

Total new parameters: 0. Architecture: unchanged. Compute: unchanged.

## Why

x0_lambdas inject raw embedding (x0) into each layer's input via `resid_lambdas[i] * x + x0_lambdas[i] * x0`. With init=0.1 and resid_lambdas=1.0, each layer adds 10% of the raw embedding. Over 8 layers, this cumulates to approximately 0.8x the raw embedding signal reaching the last layer through the residual stream. This is a substantial amount of unprocessed signal that could pollute the refined representations in deep layers.

Reducing to 0.05 (cumulative ~0.4x by layer 7):
1. Gives deep layers a cleaner residual stream with less raw embedding noise
2. Forces the model to rely more on its learned representations (attention + MLP outputs) rather than raw embeddings
3. Still preserves the critical x0 pathway (removing x0_lambdas entirely = +0.020 catastrophic)

This is the direct analog of skip_lambdas init tuning (0.0 -> 0.05, exp98) which gave -0.002 — the only other scalar init ever tuned, producing a significant improvement. With scalar_lr=0.5 and betas=(0.96, 0.95), x0_lambdas can still learn optimal values from the 0.05 starting point in 830 steps.

## Risks

- **Neutral (high)**: scalar_lr=0.5 allows rapid adaptation — the optimizer may reach the same endpoint regardless of init.
- **Regression (very low)**: 0.05 is well within the plausible optimal range. x0 remains active.
- **Crash (none)**: single constant change.

Expected range: -0.002 to +0.001.

## If it works

- Try 0.025 (even less raw embedding, test monotonicity)
- Try per-layer init (e.g., 0.1 for early layers, 0.01 for late layers — match need for raw embedding to layer depth)
- x0_lambdas init tuning is confirmed as a lever, explore further

## If it fails

- x0_lambdas=0.1 is already near-optimal, or init doesn't matter with scalar_lr=0.5
- Scalar init tuning is exhausted (skip_lambdas and x0_lambdas both tuned)
- EMBEDDING_LR=0.8 is the last untested hyperparameter (item 58)
- Remaining high-potential directions: FP8 matmuls (throughput), DeltaNet/GLA (architecture)
- 1.062436 may be near the practical ceiling for this model size and training budget


## Résultats
- val_bpb: 1.063187
- peak_vram_mb: 12503.3
- status: discarded
