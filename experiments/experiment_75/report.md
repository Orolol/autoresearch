Increase z-loss coefficient from 1e-4 to 2e-4 to strengthen logit partition function regularization (3rd attempt — exp73/74 crashed infra, never tested on GPU).

## What changed

Single number change: `z_loss = 1e-4 * logz.square().mean()` → `z_loss = 2e-4 * logz.square().mean()` (line 319 of train.py).

## Why

The z-loss coefficient has the clearest directional signal of any untested hyperparameter:
- No z-loss → z-loss 1e-4: improvement (exp9: 1.087971, kept)
- z-loss 1e-4 → z-loss 5e-5: regression (exp63: 1.080565 vs ~1.078)
- Direction: **stronger z-loss = better**. 2e-4 has never been tested on GPU.

Z-loss penalizes the log-partition function (logsumexp of logits), preventing logit magnitudes from growing unchecked. With softcap=13, the interaction is synergistic: softcap constrains the upper bound via tanh, z-loss penalizes approaching it unnecessarily, improving gradient calibration.

Exp73 and exp74 attempted this exact change but both crashed on infra (single number change = zero code risk). This is a clean retry.

## Risks

- Zero crash/OOM risk (single number change, no structural modification)
- If 2e-4 is too strong: mild regression ~0.001-0.003
- If 2e-4 is in the sweet spot: improvement ~0.0005-0.002

## If it works

- Test 3e-4 or 4e-4 (continue pushing stronger z-loss)
- Test 1.5e-4 (fine-tune if improvement is marginal)
- Combine with other architectural ideas (DeltaNet, simple MoE)

## If it fails

- 1e-4 is the optimum. Z-loss tuning is exhausted.
- Move to simple 2-expert MoE in MLP (doubles capacity at same compute)
- Try DeltaNet or Gated Linear Attention (replacing softmax attention)
