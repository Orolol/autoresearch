Reduce Muon momentum warmup from 300 to 100 steps to reach full 0.95 momentum faster in the ~830-step training budget.

## What changed
- `get_muon_momentum`: `min(step / 300, 1)` -> `min(step / 100, 1)`. Single number change.

## Why
- **36% of training is at suboptimal momentum.** With ~830 training steps and 300-step warmup, the model spends 36% of training ramping Nesterov momentum from 0.85 to 0.95. Reducing to 100 steps means full 0.95 momentum by step 100 (12% of training).
- **Higher momentum = better Nesterov look-ahead.** At step 100, the new schedule gives 0.950 momentum vs the old schedule's 0.883. The look-ahead distance is proportional to momentum/(1-momentum), so 0.950 -> distance 19 vs 0.883 -> distance 7.5. That's 2.5x stronger look-ahead for the gradient estimate.
- **NS orthogonalization provides stability.** Even with faster momentum ramp-up, the Newton-Schulz orthogonalization normalizes gradient direction, and NorMuon normalizes step size. The momentum only affects the direction weighting between current and past gradients. Inherently safe to ramp faster.
- **Never tested on GPU.** Attempted 3 times (exp38/40/49) but ALL crashed on RunPod infra. The code change is trivially correct (single number). This is the 4th attempt.

## Expected outcome
- If faster momentum ramp helps: improvement of ~0.001-0.003 (more effective gradient estimates for 24% more of training)
- If 300-step warmup was important for early stability: mild regression ~0.001-0.003
- If neutral: momentum warmup schedule doesn't matter much at this scale
- No crash, OOM, speed, or memory risk -- pure scalar change in a schedule function
