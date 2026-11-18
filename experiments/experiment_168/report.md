WEIGHT_DECAY=0.1 — AB-test lower cautious weight decay for Muon optimizer (currently 0.2, only 0.0 tested before).

## What changed

In `train.py` (line 553):
- Changed `WEIGHT_DECAY = 0.2` to `WEIGHT_DECAY = 0.1`

Total new parameters: 0. Architecture: unchanged. Compute: unchanged.

## Why

Weight decay is applied to all Muon-optimized matrix parameters via the cautious WD mechanism (only applied where gradient aligns with weight direction). The WD schedule decays linearly from WEIGHT_DECAY to 0 over training progress.

Testing history:
- WD=0.0 (exp5): 1.091936 (+0.004, clearly worse — WD helps)
- WD=0.2 (current baseline): 1.062436 (best)
- Constant WD=0.2 (exp139): 1.067816 (+0.001, decaying WD is better)
- WD=0.1 (this experiment): untested

The optimal WD between 0.0 and 0.2 has never been explored. With only ~830 training steps:
1. Parameters need to move fast to reach good solutions — less regularization helps exploration
2. The cautious WD mechanism already limits effective regularization (only applied where gradient and weight agree)
3. The linear decay schedule means WD=0.1 has half the cumulative regularization of WD=0.2
4. The model may be slightly over-regularized at WD=0.2 for this short training budget

This is explicitly listed as priority #3 on the remaining ideas in prompt.md: "WD=0.2 jamais AB-testé vs 0.1/0.3".

## Risks

- **Neutral (high)**: The delta between WD=0.1 and WD=0.2 may be within run-to-run variance (~0.003).
- **Regression (low)**: WD=0.1 is still positive WD, halfway between 0.0 (clearly bad) and 0.2 (current best). Unlikely to regress by more than 0.002.
- **Crash (none)**: Single constant change.

Expected range: -0.002 to +0.002.

## If it works

- The WD curve has its optimum below 0.2; try WD=0.05 and WD=0.15 to narrow it down
- WD is confirmed as a tunable lever with ~0.003+ impact
- Consider per-group WD (different WD for attention vs MLP matrices)

## If it fails

- WD=0.2 is at or near the optimum; try WD=0.3 to test the other direction
- If both WD=0.1 and WD=0.3 are worse, WD=0.2 is confirmed as optimal
- WD axis is then fully exhausted
- Remaining directions: FP8 matmuls (throughput), DeltaNet/GLA (architecture)
- 1.062436 is likely the practical ceiling at ~830 steps


## Résultats
- val_bpb: 1.064129
- peak_vram_mb: 12503.3
- status: discarded
