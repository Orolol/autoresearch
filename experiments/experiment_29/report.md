EMA (Exponential Moving Average) of model weights (decay=0.995) — retry of exp26/27 which both crashed on infra, never tested on GPU.

## What changed
- After `torch.compile`, clone all trainable parameters as EMA buffers
- After each optimizer step: `ema.lerp_(p.data, 1 - 0.995)` for each parameter
- Before eval: swap model parameters with EMA parameters via `p.data.copy_(ema)`
- ~6 lines of code added, zero architectural changes

## Why
- **Quality per step, not speed.** All "more steps via less compute" ideas are exhausted (MLP 3x sweet spot, DEPTH=8 minimum, torch.compile tuning failed, batch size tuning failed, factored lm_head failed). Future improvements must come from extracting more quality from the same training trajectory.
- **EMA is universally proven.** Used in diffusion models, ViTs, GANs, and LLM pretraining. It smooths out SGD noise by maintaining a running average of parameters.
- **The model is severely undertrained (~830 steps).** The final weights are noisy — observed ~0.003 variance between identical runs (exp1 run5 vs exp9). EMA averages over the last ~200 steps (decay=0.995), capturing the best of the convergence trajectory.
- **Exp26 and exp27 both crashed due to RunPod infra** (not code bugs). The idea was never tested on GPU.
- **Orthogonal to all previous experiments.** No interaction with architecture, optimizer, or schedule.

## EMA decay analysis
- `decay = 0.995`: effective averaging window ~200 steps ~24% of training
- Covers most of the warmdown phase (60% of training), smoothing the convergence trajectory
- ~152MB extra memory for BF16 model params clone (trivial vs 10GB peak VRAM)
- ~1-2ms overhead per step (one lerp_ per parameter vs ~350ms step time)

## Expected outcome
- If EMA smoothing helps: improvement of ~0.001-0.005
- If warmdown schedule already converges cleanly: neutral within noise (~0.003)
- If 0.995 averages too broadly: mild regression (~0.001-0.003)
- No crash, OOM, or speed risk


## Résultats
- val_bpb: 1.096033
- peak_vram_mb: 10451.3
- status: discarded
