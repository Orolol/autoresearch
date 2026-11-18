Retry WARMDOWN_RATIO=0.7 (exp136 crashed on infra) — longer, more gradual LR decay exploiting the strongest hyperparameter signal in the experiment history.

## What changed

One constant change in `train.py` (line 523):
- `WARMDOWN_RATIO = 0.6` -> `WARMDOWN_RATIO = 0.7`

With ~830 total steps:
- Before (0.6): ~330 steps at full LR, ~500 steps warmdown
- After (0.7): ~250 steps at full LR, ~580 steps warmdown

Zero new parameters, zero compute overhead.

## Why

- **Strongest hyperparameter signal ever observed**: exp13 showed WARMDOWN_RATIO=0.4 was +0.011 worse than 0.6. That's the largest single hyperparameter improvement in 136 experiments. The trend is clearly monotonic: more warmdown = better.
- **Never tested**: exp136 attempted 0.7 but crashed on infra (RunPod), never ran on GPU. This direction is genuinely unexplored.
- **Priority 1 exhausted**: 7 consecutive neutral/crash results (exp130: MLP 1.25x/4.75x, exp131: attention output gating, exp132: affine RMSNorm, exp133: remove post-emb norm, exp134: EMBEDDING_LR=0.4, exp135: RoPE 200x range, exp136: this same change but crashed).
- **Training dynamics lever**: WARMDOWN_RATIO affects every single step's learning rate. A longer warmdown gives the optimizer more time to slowly cool toward a better minimum.
- **Momentum/warmdown synergy**: Muon momentum warmup ends at step 300. With 0.7 warmdown, the LR starts decaying at step ~250 — smoother transition from "ramp up momentum" to "ramp down LR".

## Risks

- **Neutral (medium)**: Diminishing returns — the 0.4→0.6 jump was +0.2 with +0.011 gain. The 0.6→0.7 jump is only +0.1, so expected improvement is ~half or less.
- **Regression (low)**: Only ~250 steps at full LR (vs ~330). Rare token embeddings that update infrequently might not converge enough during the shorter constant-LR phase.
- **Crash (none)**: Pure constant change, identical to exp136 which only crashed due to infra.
- Expected range: -0.005 to +0.003

## If it works

- Try WARMDOWN_RATIO=0.8 (even more warmdown, ~166 steps at full LR, ~664 warmdown).
- The optimal warmdown may be even higher — eventually limited by the momentum warmup phase (300 steps overlap).
- This confirms the monotonic trend and opens further tuning.

## If it fails

- 0.6 is confirmed optimal for WARMDOWN_RATIO. Declare piste #30 dead.
- Remaining directions: FP8 matmuls retry (biggest speed lever), DeltaNet/GLA (complex architectural change), EMBEDDING_LR=0.8 (untested direction).


## Résultats
- val_bpb: 1.066944
- peak_vram_mb: 11984.2
- status: IMPROVED
