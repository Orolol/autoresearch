Increase FINAL_LR_FRAC from 0.1 to 0.15 — continue the winning LR floor trend from exp171, matching the modded-nanogpt PR#224 floor value.

## What changed

In `train.py` (line 557):
- Changed `FINAL_LR_FRAC = 0.1` to `FINAL_LR_FRAC = 0.15`

Total new parameters: 0. Architecture: unchanged. Compute: unchanged.

## Why

Exp171 set FINAL_LR_FRAC from 0.0 to 0.1 and achieved the new best: 1.062310 (from 1.062436). The priority queue explicitly says "Si marche: tester 0.15, 0.2."

With WARMDOWN_RATIO=0.7, the LR schedule during warmdown is:
`lr_mult = cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC`

Comparing FINAL_LR_FRAC values:
| Progress | 0.0 (old) | 0.1 (exp171) | 0.15 (this) |
|----------|-----------|--------------|-------------|
| 65% (midpoint) | 0.50 | 0.55 | 0.575 |
| 95% | 0.071 | 0.164 | 0.210 |
| 100% | 0.0 | 0.1 | 0.15 |

Average LR multiplier during warmdown: 0.50 → 0.55 → 0.575.

The modded-nanogpt PR#224 (Feb 2026) that inspired this change used a floor of exactly 0.15 with their cosine schedule. This is the exact value we're matching.

## Risks

- **Neutral (high)**: The delta between 0.1 and 0.15 is small. May be within run variance (~0.003).
- **Regression (very low)**: 0.15 of peak LR is still very conservative. Cautious WD prevents overshooting.
- **Crash (none)**: Single constant change.

Expected range: -0.001 to +0.001.

## If it works

- Try FINAL_LR_FRAC=0.2 to continue the sweep
- LR floor is a confirmed lever with clear optimum to find
- Consider applying the floor concept to other decaying schedules (WD schedule)

## If it fails

- FINAL_LR_FRAC=0.1 is the optimum; the modded-nanogpt 0.15 doesn't transfer to our linear schedule
- Try WD=0.3 (priority #3, closes the WD axis)
- LR floor axis is dead after 0.0, 0.1, 0.15
- Remaining: DeltaNet/GLA (complex architecture), Partitioned Hyperconnections


## Résultats
- val_bpb: 1.064116
- peak_vram_mb: 12503.3
- status: discarded
