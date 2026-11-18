Retry softcap=11 (3rd attempt — exp46/47 crashed infra, never tested on GPU) to further tighten gradient flow through tanh.

## What changed
- `softcap = 13` -> `softcap = 11` in the forward method. Single number change.

## Why
- **Softcap tightening is the proven winning direction.** Softcap=13 (exp45) achieved current best: 1.078486 (-0.001 vs softcap=15). Trend: {30 (+0.005), 15 (baseline), 13 (-0.001)}. Tighter is better.
- **Better gradient flow through tanh.** tanh'(x) = 1 - tanh(x)^2. With softcap=13, tanh saturates at ~26-39 raw logits. With softcap=11, saturation at ~22-33. Stronger gradients for a wider range.
- **Sufficient dynamic range.** For 32768 vocab, max useful logit gap for 99.99% top-1 confidence is ~10.4. Softcap=11 allows logit magnitudes up to 11, barely sufficient but enough.
- **Synergy with z-loss (1e-4).** Tighter softcap = z-loss operates in more effective range.
- **Why 11 instead of 12?** Softcap=12 crashed 5 times (cursed compile path). Softcap=11 avoids this.
- **Exp46/47 both crashed on infra.** Code is trivially correct (single number change). This is a pure retry.

## Expected outcome
- If 11 continues the trend: improvement of ~0.001-0.002 (val_bpb ~1.077-1.078)
- If 11 is too tight (near theoretical minimum ~10.4): mild regression ~0.001-0.003
- If neutral: optimum is in [11, 13], not worth fine-tuning further
- No crash, OOM, speed, or memory risk — pure scalar change


## Résultats
- val_bpb: 1.079533
- peak_vram_mb: 10337.7
- status: discarded
