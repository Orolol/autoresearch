EMA (Exponential Moving Average) of model weights for smoother eval — average parameters during training to reduce noise in the final model.

## What changed
- Added EMA state: clone of all model parameters at initialization
- After each optimizer step: `ema.lerp_(param, 1 - 0.995)` for each parameter
- Before eval: swap model parameters with EMA parameters
- ~6 lines of code added, zero architectural changes

## Why
- **Quality per step, not speed.** All "more steps via less compute" ideas are exhausted (MLP 3x is sweet spot, DEPTH=8 minimum, torch.compile tuning failed, batch size tuning failed). Future improvements must come from extracting more quality from the same training trajectory.
- **EMA is universally proven.** Used in diffusion models, vision transformers, GANs, and increasingly in LLM pretraining. It smooths out stochastic gradient noise by maintaining a running average of parameters.
- **The model is severely undertrained (~830 steps).** The final weights are noisy — small perturbations in the last few optimizer steps can change val_bpb by ~0.003 (observed variance between identical runs). EMA averages over the last ~200 steps (decay=0.995), capturing the best of the convergence trajectory rather than one noisy endpoint.
- **Orthogonal to all previous experiments.** No interaction with architecture, optimizer, or schedule. Pure post-processing of the weight trajectory.
- **Negligible overhead.** One `lerp_` call per parameter per step (~1-2ms vs ~350ms step time). Memory: ~220MB for the EMA buffer (model uses ~10GB VRAM).

## EMA decay analysis
- `decay = 0.995`: effective averaging window ≈ 200 steps ≈ 24% of training
- This covers most of the warmdown phase (60% of training), where the model transitions from exploration to convergence
- Too low (0.99 = 100 steps): might miss good intermediate weights during mid-warmdown
- Too high (0.999 = 1000 steps): averages include early bad weights, diluting quality

## Expected outcome
- If EMA smoothing helps (noise reduction > trajectory bias): improvement of ~0.001-0.005
- If the warmdown schedule already converges cleanly: neutral within noise (~0.003)
- If 0.995 decay averages too many weights and the mean is worse than the endpoint: mild regression (~0.001-0.003)
- No crash, OOM, or speed risk. No architectural changes. Pure eval-time improvement.
