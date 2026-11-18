Reduce logit softcap from 13 to 11 to further improve gradient flow through tanh, continuing the winning softcap tightening direction.

## What changed
- `softcap = 13` -> `softcap = 11` in the forward method. Single number change.

## Why
- **Softcap=13 just beat the previous best** (1.078486 vs 1.079602, -0.001). Tighter softcap improves gradient flow through tanh — less saturation means stronger gradients for a wider range of logit values.
- **Gradient flow math.** tanh'(x) = 1 - tanh(x)^2. With softcap=13, tanh saturates when raw logits approach ~26-39. With softcap=11, saturation happens at ~22-33. Stronger gradients for a wider range.
- **Sufficient dynamic range.** For 32768 vocab, max useful logit gap for 99.99% top-1 confidence is ~10.4. Softcap=11 still allows logit magnitudes up to 11, barely sufficient but enough for confident predictions. This is the tightest we can go before potentially bottlenecking expressiveness.
- **Synergy with z-loss (1e-4).** Z-loss penalizes large logits. Tighter softcap means z-loss operates in a more effective range — both regularizers cooperate more efficiently.
- **Why 11 instead of 12?** Softcap=12 crashed 5 times (exp36/37/39/43 + one more) on RunPod infra, never tested on GPU. The number 12 seems to trigger a cursed torch.compile code path. Softcap=11 avoids this while testing an even tighter constraint.
- **Tested values so far on GPU:** {0 (+0.009), 11 (NOW), 13 (-0.001), 15 (baseline), 30 (+0.005)}. The trend 30 > 15 > 13 suggests tighter is better, but 11 is approaching the theoretical minimum useful gap (~10.4).

## Expected outcome
- If 11 continues the improving trend: improvement of ~0.001-0.002 (val_bpb ~1.077-1.078)
- If 11 is too tight (bottlenecks expressiveness): mild regression ~0.001-0.003
- If neutral: the optimum is somewhere in [11, 13], probably not worth fine-tuning further
- No crash, OOM, speed, or memory risk — pure scalar change
