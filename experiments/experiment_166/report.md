Increase EMBEDDING_LR from 0.6 to 0.8 for faster token embedding convergence in the largest parameter group (22% of model).

## What changed

In `train.py` (line 549):
- Changed `EMBEDDING_LR = 0.6` to `EMBEDDING_LR = 0.8`

Total new parameters: 0. Architecture: unchanged. Compute: unchanged.

## Why

The embedding learning rate controls the wte (word token embedding) parameters — 16.8M params, 22% of the total model. This is the single largest parameter group and one of the least tuned hyperparameters.

With `dmodel_lr_scale = (512/768)^-0.5 = 1.22`, the effective embedding LR goes from 0.6 × 1.22 = 0.73 to 0.8 × 1.22 = 0.98. Higher LR helps embeddings converge faster within the ~830-step training budget.

Testing history:
- 0.4 (exp134): 1.067798, neutral (+0.0007 vs baseline)
- 0.6 (current): produces best results (1.062436)
- 0.8 (this experiment): untested

The trend from 0.4→0.6 was positive (0.4 was slightly worse). Testing 0.8 explores whether this trend continues. With Adam betas=(0.8, 0.95) and no weight decay on embeddings, the optimizer has sufficient adaptive control to handle higher LR without instability.

All simple architectural changes have been exhausted over 165 experiments. The remaining high-potential directions (FP8 matmuls, DeltaNet/GLA) require multi-experiment implementation effort. EMBEDDING_LR=0.8 is the last explicitly listed untested simple idea (item 58 in prompt.md).

## Risks

- **Neutral (high)**: The difference between 0.6 and 0.8 may be too small to measure given ~0.003 run variance.
- **Regression (low)**: Effective LR=0.98 is aggressive but within Adam's adaptive range. If too high, the optimizer's per-parameter step sizing compensates.
- **Crash (none)**: Single constant change.

Expected range: -0.001 to +0.001.

## If it works

- Try 1.0 (effective 1.22) to test monotonicity
- The embedding LR axis is confirmed as a lever; explore further with 0.9, 1.0
- Consider separate VE LR tuning (currently same as embedding LR)

## If it fails

- 0.6 is confirmed as optimal embedding LR (0.4 neutral, 0.8 neutral or worse)
- Embedding LR axis is dead
- All simple hyperparameters are fully tuned
- Remaining directions: FP8 matmuls (complex, crashed in exp121), DeltaNet/GLA (complex, multi-experiment)
- 1.062436 may be near the practical ceiling for this model size and ~830-step training budget


## Résultats
- val_bpb: 1.062787
- peak_vram_mb: 12503.3
- status: discarded
