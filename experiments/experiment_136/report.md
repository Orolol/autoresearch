Increase WARMDOWN_RATIO from 0.6 to 0.7 for longer, more gradual LR decay.

## What changed

One constant change in `train.py` (line 523):
- `WARMDOWN_RATIO = 0.6` -> `WARMDOWN_RATIO = 0.7`

With ~830 total steps:
- Before (0.6): ~330 steps at full LR, ~500 steps warmdown
- After (0.7): ~250 steps at full LR, ~580 steps warmdown

Zero new parameters, zero compute overhead.

## Why

- **Strongest untested hyperparameter trend**: exp13 showed WARMDOWN_RATIO=0.4 was +0.011 worse than 0.6. That's the largest single hyperparameter signal in the entire history. The trend is clearly monotonic: more warmdown = better. 0.7 has never been tested.
- **Priority 1 exhausted**: 6 consecutive neutral architectural experiments (exp130: MLP 1.25x/4.75x, exp131: attention output gating, exp132: affine RMSNorm, exp133: remove post-emb norm, exp134: EMBEDDING_LR=0.4, exp135: RoPE 200x range). All within noise of best 1.067071.
- **Training dynamics lever**: WARMDOWN_RATIO affects every single step. A longer warmdown gives the optimizer more time to slowly cool toward a better minimum, which is especially valuable with Muon's momentum dynamics.
- **Momentum/warmdown synergy**: Muon momentum warmup ends at step 300. With 0.7 warmdown, the LR starts decaying at step ~250 — the transition from "ramp up momentum" to "ramp down LR" is smoother.

## Risks

- **Neutral (medium)**: Diminishing returns — the 0.4→0.6 jump was +0.2 with +0.011 gain. The 0.6→0.7 jump is only +0.1, so expected improvement is smaller, possibly within noise.
- **Regression (low)**: Only ~250 steps at full LR (vs ~330). Rare token embeddings that update infrequently might not converge enough during the shorter constant-LR phase.
- **Crash (none)**: Pure constant change.
- Expected range: -0.005 to +0.003

## If it works

- Try WARMDOWN_RATIO=0.8 (even more warmdown, ~166 steps at full LR, ~664 warmdown).
- The optimal warmdown may be even higher — eventually limited by the momentum warmup phase.
- This would be the first hyperparameter improvement since skip_lambdas init=0.05 (exp98).

## If it fails

- 0.6 is confirmed optimal (or near-optimal) for WARMDOWN_RATIO. Declare piste #30 dead.
- Remaining directions: FP8 matmuls retry (biggest speed lever), DeltaNet/GLA (complex architectural change).
- Consider trying EMBEDDING_LR=0.8 (only 0.4 tested and was neutral, but 0.8 pushes the other direction).
