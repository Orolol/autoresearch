Reduce logit softcap from 15 to 13 to improve gradient flow through tanh while preserving sufficient logit dynamic range.

## What changed
- `softcap = 15` -> `softcap = 13` in the forward method. Single number change.

## Why
- **Softcap is the most impactful hyperparameter after batch size.** Removing it entirely caused +0.009 regression. Softcap=30 caused +0.005 regression. But only 3 coarse values tested on GPU: {0, 15, 30}. The 10-20 range has never been explored on GPU.
- **Better gradient flow through tanh.** With softcap=15, tanh saturates when raw logits approach ~30-45. With softcap=13, saturation happens at ~26-39. Tighter constraint = stronger gradients for a wider range of logit values. tanh'(x) = 1 - tanh(x)^2: smaller x -> larger derivative -> better backprop signal.
- **Synergy with z-loss (1e-4).** Z-loss already penalizes large logits. Tighter softcap means z-loss has less work to do — both regularizers cooperate more efficiently.
- **Sufficient dynamic range.** For 32768 vocab, max useful logit gap for 99.99% top-1 confidence is ~10.4. Softcap=13 allows logit magnitudes up to 13, sufficient for even highly confident predictions. More headroom than softcap=12.
- **Why 13 instead of 12?** Softcap=12 was attempted 5 times (exp36/37/39/43 + one more) and crashed EVERY time on RunPod infra, never tested on GPU. The different constant value may avoid a torch.compile code-path issue that makes 12 unstable. 13 is a compromise between 12 (never tested) and 15 (baseline).

## Expected outcome
- If 13 improves gradient flow without hurting expressiveness: improvement of ~0.001-0.003
- If 15 was already optimal: neutral within noise (~0.003)
- If 13 is too restrictive: mild regression (~0.001-0.003)
- No crash, OOM, speed, or memory risk — pure scalar change


## Résultats
- val_bpb: 1.078486
- peak_vram_mb: 10337.7
- status: IMPROVED
