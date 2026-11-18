Reduce TOTAL_BATCH_SIZE from 262144 to 196608 (grad_accum 4 to 3) for 33% more optimizer steps with only 15% noisier gradients.

## What changed

Single constant change: `TOTAL_BATCH_SIZE = 2**18` (262144) -> `TOTAL_BATCH_SIZE = 196608` (3 * 65536).

This reduces gradient accumulation from 4 micro-steps to 3, giving 33% more optimizer steps in the 5-minute budget.

## Why

The strongest known lever in this series is "more optimizer steps." The biggest single improvement was MLP 3x (exp14, -0.008), which worked purely because reducing compute per step gave more steps in the fixed time budget.

Batch size is another axis for the same tradeoff:
- 2^18 = 262144 (current): grad_accum=4, baseline
- 196608 (this experiment): grad_accum=3, 33% more steps, 15% more gradient noise
- 2^17 = 131072 (exp6): grad_accum=2, 100% more steps, 41% more noise -> +0.011 REGRESSION

The 2^17 result showed that noise dominated at 41% increase. But 196608 is a never-tested intermediate point where the noise increase (15%) is much smaller and the step gain (33%) is still substantial. The noise-steps tradeoff is nonlinear, and this midpoint could be in the zone where step gains outweigh noise cost.

Math:
- tokens_per_fwdbwd = 32 * 2048 = 65536
- 196608 / 65536 = 3 (clean division, no code changes beyond the constant)
- Gradient noise: sqrt(262144/196608) = 1.155x (15% noisier)
- Steps: 262144/196608 = 1.333x (33% more steps)

## Risks

- Zero crash/OOM risk (single number change, smaller batch = less memory)
- If gradients still too noisy: regression ~0.002-0.005 (informs batch size curve)
- If step gain dominates: improvement ~0.001-0.003

## If it works

- Try 229376 (grad_accum=3.5... no, must be integer). Try 3.5 * 65536 = not integer. So the next point is grad_accum=3 (this) vs grad_accum=4 (current). If this works, we already have the optimum at 3.
- Combine with z-loss 2e-4 (clear directional signal from exp63: more z-loss is better)

## If it fails

- 262144 is confirmed as the batch size sweet spot. Batch size tuning is fully dead.
- Move to z-loss 2e-4 (stronger z-loss, following the 5e-5 < 1e-4 signal from exp63)
- Move to Schedule-Free Adam for AdamW groups only
- Consider FP8 matmuls on Blackwell (last "more steps" frontier via faster compute)


## Résultats
- val_bpb: 1.090227
- peak_vram_mb: 10337.7
- status: discarded
