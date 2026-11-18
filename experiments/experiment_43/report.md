Reduce logit softcap from 15 to 12 (5th attempt — exp36/37/39 all crashed on infra, never tested on GPU) for better gradient flow through tanh.

## What changed
- `softcap = 15` -> `softcap = 12` in the forward method. Single number change.

## Why
- **Softcap is the most impactful hyperparameter after batch size.** Removing it entirely caused +0.009 regression (exp11 old series). Softcap=30 caused +0.005 regression. But only 3 coarse values tested: {0, 15, 30}. The 10-20 range has never been explored.
- **Better gradient flow through tanh.** With softcap=15, tanh saturates when raw logits approach ~30-45. With softcap=12, saturation happens at ~24-36. Tighter constraint = stronger gradients for a wider range of logit values. tanh'(x) = 1 - tanh(x)^2: smaller x → larger derivative → better backprop signal.
- **Synergy with z-loss (1e-4).** Z-loss already penalizes large logits. Tighter softcap means z-loss has less work to do — both regularizers cooperate more efficiently.
- **Sufficient dynamic range.** For 32768 vocab, max useful logit gap for 99.99% top-1 confidence is ~10.4. Softcap=12 allows logit magnitudes up to 12, sufficient for even highly confident predictions.
- **4 previous attempts all crashed on RunPod infra** (exp36/37/39 + one more). Code was verified correct each time — the crashes were CUDA driver init failures and API timeouts, not code bugs. This is the 5th attempt.

## Expected outcome
- If 12 improves gradient flow without hurting expressiveness: improvement of ~0.001-0.003
- If 15 was already optimal: neutral within noise (~0.003)
- If 12 is too restrictive: mild regression (~0.001-0.003)
- No crash, OOM, speed, or memory risk — pure scalar change
