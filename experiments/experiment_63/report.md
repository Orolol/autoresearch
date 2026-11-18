Reduce z-loss coefficient from 1e-4 to 5e-5 to halve partition function regularization strength.

## What changed

Single number change: `z_loss = 1e-4 * logz.square().mean()` -> `z_loss = 5e-5 * logz.square().mean()` (line 306 of train.py).

## Why

Z-loss was introduced in exp9 and has been kept since, but its coefficient has **never been tuned**. The coefficient 1e-4 was the default PaLM value. With softcap=13 already constraining logits to [-13, 13], the z-loss is partially redundant:

1. **Softcap already limits logit magnitude.** The tanh squashing prevents logits from growing large, which is z-loss's primary job (penalizing large partition functions). With both mechanisms active, z-loss may be adding unnecessary gradient noise.

2. **Gradient bandwidth.** At 1e-4, the z-loss gradient competes with the cross-entropy gradient. Reducing to 5e-5 preserves the stability benefit (catching rare outlier logit configurations) while reducing interference with the primary objective.

3. **Conservative direction.** Rather than removing z-loss entirely (which would lose the stability benefit that helped in exp9), halving the coefficient tests whether less regularization is better. If the softcap is sufficient for stability, we free ~50% of the z-loss gradient magnitude for actual loss minimization.

## Risks

- Zero crash/OOM/speed risk (no new parameters, no structural changes)
- If 5e-5 is too weak: z-loss becomes near-irrelevant, possible mild instability late in training (~0.001-0.003 regression)
- If 1e-4 was already optimal: no meaningful change (within noise)


## Résultats
- val_bpb: 1.080565
- peak_vram_mb: 10337.7
- status: discarded
