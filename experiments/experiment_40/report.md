Retry Muon momentum warmup 300→100 steps (exp38 crashed infra, never tested on GPU) to reach full 0.95 momentum faster.

## What changed
- `get_muon_momentum`: `min(step / 300, 1)` → `min(step / 100, 1)`. Single number change.
- Momentum still ramps from 0.85 to 0.95, but reaches 0.95 at step 100 instead of step 300.

## Why
- **36% of training at suboptimal momentum.** The model trains ~830 steps total. With a 300-step warmup, the optimizer spends 300/830 = 36% of training ramping momentum from 0.85 to 0.95. During this time, Nesterov look-ahead is weaker, reducing effective step size and exploration.
- **100-step warmup = 12% at suboptimal momentum.** Cuts the warmup fraction by 3×. The optimizer reaches full 0.95 momentum by step 100, giving ~730 steps (88% of training) at optimal momentum instead of ~530 steps (64%).
- **The momentum range is small (0.85→0.95).** A 10% difference in momentum is mild. Even at 0.85, the optimizer is functional. NS orthogonalization provides inherent stability regardless of momentum magnitude.
- **Nesterov momentum benefits from full strength early.** Higher momentum = better look-ahead gradient at `θ + momentum * velocity`. At step 100, new schedule has 0.950 vs 0.883 momentum (3.5× stronger look-ahead distance: 19 vs 7.5).
- **Exp38 crashed on RunPod infra — code was correct.** This is a pure retry. The change is a single number in a scalar computation, zero risk of code issues.

## Expected outcome
- If faster warmup helps convergence: improvement of ~0.001-0.003
- If 300-step warmup was already fine: neutral within noise (~0.003)
- If 100-step warmup causes early instability: mild regression (~0.001-0.003)
- No crash, OOM, speed, or memory risk — pure schedule scalar change
