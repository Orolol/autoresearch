LR Floor — set FINAL_LR_FRAC=0.1 to prevent learning rate from decaying to zero during warmdown, inspired by modded-nanogpt PR #224 (Feb 2026).

## What changed

In `train.py` (line 557):
- Changed `FINAL_LR_FRAC = 0.0` to `FINAL_LR_FRAC = 0.1`

Total new parameters: 0. Architecture: unchanged. Compute: unchanged.

## Why

With WARMDOWN_RATIO=0.7 and FINAL_LR_FRAC=0.0, the LR schedule is:
- 0%-30% progress: full LR (multiplier = 1.0)
- 30%-100% progress: linear decay from 1.0 to **0.0**

The formula: `cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC` where `cooldown = (1 - progress) / 0.7`.

At 95% progress: LR multiplier = 0.071. At 99%: 0.014. The last ~50 steps (out of ~830) have LR below 0.06 and contribute almost nothing to learning. These are essentially wasted compute.

With FINAL_LR_FRAC=0.1:
- At 95% progress: multiplier = 0.071 + 0.929*0.1 = 0.164 (2.3x higher)
- At 100% progress: multiplier = 0.1 (instead of 0.0)
- Midpoint (65% progress): multiplier = 0.55 (instead of 0.50)

This was directly inspired by the modded-nanogpt speedrun PR #224 (Feb 2026), which introduced an LR floor of 0.15 and achieved statistically significant improvement across 15 runs (p=0.0001). Their setup is different (GPT-2 124M, 8xH100, cosine schedule), but the principle applies: preventing near-zero LR keeps final steps productive.

The average LR multiplier during warmdown increases from 0.50 to 0.55 — a 10% increase in effective learning. This is equivalent to gaining ~40 effective full-LR steps out of the ~830 total.

## Risks

- **Neutral (medium)**: The effect is modest — ~10% higher average LR during warmdown. The delta may be within noise (~0.003).
- **Regression (low)**: Higher LR at the end could prevent fine convergence. But 10% of peak LR is still very low, and the cautious WD mechanism prevents overshooting.
- **Crash (none)**: Single constant change.

Expected range: -0.002 to +0.001.

## If it works

- Try FINAL_LR_FRAC=0.15 and 0.2 to find the optimum
- Combine with WD=0.3 (the other untested hyperparameter on the priority queue)
- The "LR floor" principle could apply to other schedules (e.g., scalar_lr, WD schedule)

## If it fails

- FINAL_LR_FRAC=0.0 is confirmed correct; the final steps being "wasted" is acceptable
- Try WD=0.3 (priority #3 on queue, simple, informative)
- FP8 is dead (4+ crashes); DeltaNet/GLA hybrid is the next architectural frontier
- The model is near its practical ceiling (~1.062) at ~830 steps


## Résultats
- val_bpb: 1.062310
- peak_vram_mb: 12503.3
- status: IMPROVED
