Increase WARMDOWN_RATIO from 0.7 to 0.8 — continuing the strongest monotonic hyperparameter trend in 137 experiments.

## What changed

One constant change in `train.py` (line 523):
- `WARMDOWN_RATIO = 0.7` -> `WARMDOWN_RATIO = 0.8`

With ~830 total steps:
- Before (0.7): ~250 steps at full LR, ~580 steps warmdown
- After (0.8): ~166 steps at full LR, ~664 steps warmdown

Zero new parameters, zero compute overhead.

## Why

- **Confirmed monotonic trend across 3 data points**: 0.4 was +0.011 worse than 0.6 (exp13), 0.7 beat 0.6 by -0.000127 (exp137, new best 1.066944). Each increase in warmdown has improved results. 0.8 is the natural next point.
- **Explicitly recommended**: exp137 report says "Try WARMDOWN_RATIO=0.8 (even more warmdown, ~166 steps at full LR, ~664 warmdown)."
- **Priority 1 exhausted**: 8+ consecutive neutral/crash architectural experiments (exp130-135, plus earlier). All known architectural changes are dead or too complex (DeltaNet/GLA).
- **Longer warmdown = smoother optimization**: More gradual LR decay gives the optimizer more time to settle into sharper minima, especially valuable with Muon's momentum dynamics.
- **Momentum overlap**: Muon momentum warmup ends at step 300. With 0.8 warmdown, LR starts decaying at step ~166 — the optimizer is still building momentum while LR is already declining. This was already the case at 0.7 (warmdown start ~250) and it worked.

## Risks

- **Neutral (medium-high)**: Diminishing returns. The 0.4→0.6 jump (+0.2) gave -0.011. The 0.6→0.7 jump (+0.1) gave -0.000127 — already 85x smaller gain per unit increase. The 0.7→0.8 jump may be deep into diminishing returns territory.
- **Regression (low-medium)**: Only ~166 steps at full LR. Sparse embedding updates (wte, VE) may suffer. Muon momentum warmup (300 steps) now starts before warmdown — the optimizer never reaches peak "everything at full power" state.
- **Crash (none)**: Pure constant change.
- Expected range: -0.003 to +0.005

## If it works

- Try WARMDOWN_RATIO=0.9 (even more aggressive — only ~83 steps at full LR).
- Consider that the "optimal" warmdown might approach 1.0, meaning the model benefits most from continuously declining LR from the start.
- Update prompt.md with the new optimum.

## If it fails

- 0.7 is the optimum. Declare WARMDOWN_RATIO tuning done.
- Remaining directions: FP8 matmuls retry (biggest speed lever), DeltaNet/GLA (complex but highest potential), EMBEDDING_LR=0.8 (untested high direction).
- Consider that the short constant-LR phase hurts sparse embedding convergence. Possible mitigation: separate warmdown schedule for AdamW vs Muon groups.


## Résultats
- val_bpb: 1.067077
- peak_vram_mb: 11984.2
- status: discarded
