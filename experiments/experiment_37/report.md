Retry softcap=12 (exp36 crashed on infra, never tested on GPU) to improve gradient flow through tanh.

## What changed
- `softcap = 15` -> `softcap = 12` in the forward method. Single number change. Identical to exp36.

## Why
- **Exp36 crashed on RunPod infra** — the code was correct but never ran on GPU. This is a pure retry.
- **Softcap is one of the most impactful hyperparameters.** Removing it entirely caused +0.009 regression (exp11 old series). Softcap=30 caused +0.005 regression (exp2). But we've only tested 3 coarse values: {0, 15, 30}. The optimal might be slightly different from 15.
- **Better gradient flow through tanh.** With softcap=15, tanh saturates when raw logits approach ~30-45. With softcap=12, saturation happens at ~24-36. Tighter constraint = stronger gradients for a wider range of logit values. tanh'(x) = 1-tanh(x)^2, and smaller x means larger derivative.
- **Synergy with z-loss (1e-4).** Z-loss already penalizes large logits. Tighter softcap means z-loss has less work to do — both regularizers work together more efficiently.
- **Sufficient dynamic range.** For 32768 vocab, max useful logit gap for 99.99% top-1 confidence is ~10.4. Softcap=12 allows logit magnitudes up to 12, sufficient for even highly confident predictions.
- **Fine-grained search.** The 3-point search {0, 15, 30} is very coarse. Testing 12 narrows toward the optimal.

## Expected outcome
- If 12 improves gradient flow without hurting expressiveness: improvement of ~0.001-0.003
- If 15 was already optimal: neutral within noise (~0.003)
- If 12 is too restrictive: mild regression (~0.001-0.003)
- No crash, OOM, speed, or memory risk — pure scalar change
