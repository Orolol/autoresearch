Reduce logit softcap from 15 to 12 to improve gradient flow through tanh while preserving sufficient logit dynamic range.

## What changed
- `softcap = 15` -> `softcap = 12` in the forward method. Single number change.

## Why
- **Softcap is one of the most impactful hyperparameters.** Removing it entirely (softcap=0) caused +0.009 regression. Softcap=30 caused +0.005 regression. But we've only tested 3 coarse values: 0, 15, 30. The optimal might be slightly different from 15.
- **Better gradient flow through tanh.** With softcap=15, the tanh saturates when raw logits approach ~30-45. With softcap=12, saturation happens at ~24-36. A tighter constraint means the gradient through `softcap * tanh(logits / softcap)` is stronger for a wider range of logit values — tanh'(x) = 1-tanh(x)^2, and smaller x means larger derivative.
- **Synergy with z-loss.** Z-loss (1e-4) already penalizes large logits. A tighter softcap means z-loss has less work to do — the two regularizers work together more efficiently.
- **Sufficient dynamic range.** For a vocab of 32768 tokens, the maximum useful logit gap for 99.99% top-1 confidence is ~10.4. Softcap=12 allows logit magnitudes up to 12, which is sufficient for even highly confident predictions. The model doesn't need logits > 12 for any practical purpose.
- **Fine-grained search.** The 3-point search {0, 15, 30} is very coarse. If the optimal is 12 or 10, we'd never find it without testing. This experiment narrows the search toward the optimal softcap value.

## Expected outcome
- If 12 improves gradient flow without hurting expressiveness: improvement of ~0.001-0.003
- If 15 was already optimal: neutral within noise (~0.003)
- If 12 is too restrictive for some predictions: mild regression (~0.001-0.003)
- No crash, OOM, speed, or memory risk — pure scalar change in the loss computation
