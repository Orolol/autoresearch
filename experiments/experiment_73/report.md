Increase z-loss coefficient from 1e-4 to 2e-4 to strengthen logit partition function regularization, following the clear directional signal from exp63.

## What changed

Single number change: `z_loss = 1e-4 * logz.square().mean()` → `z_loss = 2e-4 * logz.square().mean()` (line 319 of train.py).

## Why

The z-loss coefficient has a clear directional signal that has never been fully exploited:
- Exp9: z-loss at 1e-4 was an improvement over no z-loss (1.087971, kept)
- Exp63: z-loss at 5e-5 was worse than 1e-4 by +0.002 (1.080565 vs ~1.078)
- Direction: **stronger z-loss = better**. 2e-4 has never been tested.

Z-loss penalizes the log-partition function (logsumexp of logits), preventing logit magnitudes from growing unchecked. With softcap=13, logits are already bounded by tanh, but the z-loss gradient provides additional signal that helps the model learn better calibrated predictions. The interaction between z-loss and softcap is synergistic: softcap constrains the upper bound, z-loss penalizes approaching it unnecessarily.

Doubling the coefficient from 1e-4 to 2e-4 is a conservative increase that maintains the established direction. The auxiliary loss contribution doubles from ~0.001-0.002 to ~0.002-0.004 (still negligible vs main CE loss of ~3.0-4.0).

## Risks

- Zero crash/OOM risk (single number change, no structural modification)
- If 2e-4 is too strong: mild regression ~0.001-0.003 (logit gradient dominated by regularization)
- If 2e-4 is in the sweet spot: improvement ~0.0005-0.002 (better calibrated logits = lower val_bpb)
- If the optimum is between 1e-4 and 2e-4: try 1.5e-4 next
