Retry softcap=12 (3rd attempt — exp36/37 crashed on infra, never tested on GPU) to improve gradient flow through tanh.

## What changed
- `softcap = 15` -> `softcap = 12` in the forward method. Single number change. Identical to exp36/37.

## Why
- **Softcap is one of the most impactful hyperparameters.** Removing it entirely caused +0.009 regression. Softcap=30 caused +0.005 regression. But only 3 coarse values tested: {0, 15, 30}. The optimal might be near 12.
- **Better gradient flow through tanh.** With softcap=15, tanh saturates when raw logits approach ~30-45. With softcap=12, saturation happens at ~24-36. Tighter constraint = stronger gradients for a wider range of logit values. tanh'(x) = 1-tanh(x)^2, smaller x = larger derivative.
- **Synergy with z-loss (1e-4).** Z-loss already penalizes large logits. Tighter softcap means z-loss has less work to do — both regularizers work together more efficiently.
- **Sufficient dynamic range.** For 32768 vocab, max useful logit gap for 99.99% top-1 confidence is ~10.4. Softcap=12 allows logit magnitudes up to 12, sufficient for even highly confident predictions.
- **Exp36 and exp37 both crashed on RunPod infra** — code was correct, never ran on GPU. This is the 3rd attempt at the same single-number change.

## Expected outcome
- If 12 improves gradient flow without hurting expressiveness: improvement of ~0.001-0.003
- If 15 was already optimal: neutral within noise (~0.003)
- If 12 is too restrictive: mild regression (~0.001-0.003)
- No crash, OOM, speed, or memory risk — pure scalar change
