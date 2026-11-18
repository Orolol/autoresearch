WEIGHT_DECAY=0.3 — AB-test higher cautious weight decay for Muon optimizer to close the WD optimization axis.

## What changed

In `train.py` (line 553):
- Changed `WEIGHT_DECAY = 0.2` to `WEIGHT_DECAY = 0.3`

Total new parameters: 0. Architecture: unchanged. Compute: unchanged.

## Why

The weight decay axis has never been fully explored on the high side:
- WD=0.0: 1.092 (+0.004 vs best at the time, exp5)
- WD=0.1: 1.064 (+0.002 vs best at the time, exp168)
- WD=0.2: current best (1.062310)
- WD=0.3: **untested**

The trend is monotonically improving toward higher WD. This experiment determines whether 0.3 continues the trend or 0.2 is the peak.

Key context: Muon uses "cautious" weight decay — it only applies WD to weights where the gradient and parameter have the same sign (`mask = (g * params) >= 0`). This means 0.3 is less aggressive than it sounds: the masking prevents decay from fighting the gradient direction. Higher WD under cautious masking primarily strengthens regularization of aligned weights.

Priority #1 (LR floor) is now closed: FINAL_LR_FRAC=0.1 is optimal (exp171 improved, exp172 at 0.15 regressed). This is the next item on the priority queue (#3).

## Risks

- **Regression (medium)**: Training is only 5 minutes (~830 steps). Higher WD could over-regularize, preventing the model from reaching full capacity. The cautious masking mitigates this but doesn't eliminate it.
- **Neutral (medium)**: The delta between 0.2 and 0.3 may be within run variance (~0.003).
- **Crash (none)**: Single constant change.

Expected range: -0.002 to +0.003.

## If it works

- Try WD=0.4 to continue the sweep
- The WD curve would suggest the optimum is at 0.3+, which could mean even more aggressive regularization helps

## If it fails

- WD=0.2 is confirmed optimal; the axis is dead
- Remaining priorities: DeltaNet/GLA hybrid (#2, complex), Partitioned Hyperconnections (#4, complex)
- The model is near its practical ceiling (~1.062) at ~830 steps
