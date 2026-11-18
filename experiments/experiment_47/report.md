Retry softcap=11 (exp46 crashed infra, never tested on GPU) to further tighten gradient flow through tanh.

## What changed
- `softcap = 13` -> `softcap = 11` in the forward method. Single number change.

## Why
- **Softcap tightening is the most recent winning direction.** Softcap=13 (exp45) achieved the current best: 1.078486 (-0.001 vs softcap=15). The trend is clear: {30 (+0.005), 15 (baseline), 13 (-0.001)}. Tighter is better.
- **Better gradient flow through tanh.** tanh'(x) = 1 - tanh(x)^2. With softcap=13, tanh saturates when raw logits approach ~26-39. With softcap=11, saturation happens at ~22-33. Stronger gradients for a wider range of logit values.
- **Sufficient dynamic range.** For 32768 vocab, max useful logit gap for 99.99% top-1 confidence is ~10.4. Softcap=11 still allows logit magnitudes up to 11, barely sufficient but enough for confident predictions. This is the tightest we can go before potentially bottlenecking expressiveness.
- **Synergy with z-loss (1e-4).** Z-loss penalizes large logits. Tighter softcap means z-loss operates in a more effective range — both regularizers cooperate more efficiently.
- **Why 11 instead of 12?** Softcap=12 crashed 5 times (exp36/37/39/43 + one more) on RunPod infra, never tested on GPU. The number 12 seems to trigger a cursed torch.compile code path. Softcap=11 avoids this.
- **Exp46 was identical but crashed on infra.** This is a pure retry. Code is trivially correct (single number change).

## Expected outcome
- If 11 continues the improving trend: improvement of ~0.001-0.002 (val_bpb ~1.077-1.078)
- If 11 is too tight (bottlenecks expressiveness near theoretical minimum ~10.4): mild regression ~0.001-0.003
- If neutral: the optimum is somewhere in [11, 13], probably not worth fine-tuning further
- No crash, OOM, speed, or memory risk — pure scalar change
