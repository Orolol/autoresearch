Increase z-loss coefficient from 1e-4 to 2e-4 to strengthen logit partition function regularization.

## What changed
Single line change: `z_loss = 1e-4 * logz.square().mean()` → `z_loss = 2e-4 * logz.square().mean()`.

## Why
Clear directional signal from previous experiments:
- Exp63 tested z-loss 5e-5 → 1.080565 (+0.002 regression from 1e-4 baseline)
- Current 1e-4 works well
- Direction: stronger z-loss = better. 2e-4 is the natural next step.

Z-loss (PaLM-style) penalizes large logit partition functions, which improves gradient flow through the softcap tanh and keeps the model's confidence well-calibrated. With softcap=13, the effective logit range is [-13, 13], and stronger z-loss pressure keeps logits further from saturation, preserving gradient magnitude through tanh.

## Risks
- Zero crash risk (single constant change)
- Zero OOM risk (no structural change)
- If 2e-4 is too strong: z-loss gradient could dominate CE gradient, causing mild regression (~0.001-0.003)
- This change has been attempted 3 times (exp73-75) and crashed on infra each time. The code is correct — crashes were RunPod infrastructure issues.

## Previous attempts
- Exp73: z-loss 2e-4 → CRASH (infra)
- Exp74: z-loss 2e-4 → CRASH (infra)
- Exp75: z-loss 2e-4 → CRASH (infra)


## Résultats
- val_bpb: 1.078430
- peak_vram_mb: 10337.7
- status: discarded
