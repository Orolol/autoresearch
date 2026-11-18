Reduce Muon momentum warmup from 300 to 100 steps to reach full momentum faster in the ~830-step training budget.

## What changed
- `get_muon_momentum`: `min(step / 300, 1)` → `min(step / 100, 1)`. Single number change.
- Momentum still ramps from 0.85 to 0.95, but reaches 0.95 at step 100 instead of step 300.

## Why
- **36% of training at suboptimal momentum.** The model trains ~830 steps total. With a 300-step warmup, the optimizer spends 300/830 = 36% of training ramping momentum from 0.85 to 0.95. During this time, Nesterov momentum is weaker, reducing the effective step size and exploration ability.
- **100-step warmup = 12% at suboptimal momentum.** Cuts the warmup fraction by 3×. The optimizer reaches full 0.95 momentum by step 100, giving ~730 steps (88% of training) at optimal momentum instead of ~530 steps (64%).
- **The momentum range is small (0.85→0.95).** A 10% difference in momentum is mild. Even at 0.85, the optimizer is functional. The risk of instability from faster warmup is minimal — the model already has QK-norm, softcap=15, and z-loss for stability.
- **Nesterov momentum benefits from full strength early.** Nesterov uses the look-ahead gradient at `θ + momentum * velocity`. Higher momentum = better look-ahead = more accurate gradient direction. Getting to 0.95 sooner means better gradient estimates for 200 more steps.
- **NS orthogonalization provides inherent stability.** The Newton-Schulz step normalizes gradient direction regardless of momentum magnitude. Higher momentum just means larger step magnitudes, which NorMuon's variance reduction handles adaptively.
- **Never tested.** All 37 previous experiments used the default 300-step momentum warmup. This is the first Muon momentum schedule experiment.

## Momentum comparison at key steps
| Step | 300-step warmup | 100-step warmup |
|------|----------------|-----------------|
| 0    | 0.850          | 0.850           |
| 50   | 0.867          | 0.900           |
| 100  | 0.883          | 0.950           |
| 200  | 0.917          | 0.950           |
| 300  | 0.950          | 0.950           |

At step 100, the new schedule has 7.6% higher momentum (0.950 vs 0.883). This translates to ~3.5× stronger Nesterov look-ahead (the look-ahead distance scales as momentum/(1-momentum): 19 vs 7.5).

## Expected outcome
- If faster warmup helps convergence: improvement of ~0.001-0.003
- If 300-step warmup was already fine: neutral within noise (~0.003)
- If 100-step warmup causes early instability: mild regression (~0.001-0.003), can retry with 150 or 200
- No crash, OOM, speed, or memory risk — pure schedule scalar change
