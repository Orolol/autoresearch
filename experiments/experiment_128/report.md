V-only token shift — remove K shift from exp126, keep only V shift to avoid blurring key position identity before RoPE.

## What changed

Simplified token shift in `CausalSelfAttention` from K+V to V-only:

1. **`__init__`**: Removed `self.time_mix_k`. Only `self.time_mix_v` remains.

2. **`forward`**: K projection now uses unshifted `x` (same as original pre-exp126):
   - `k = self.c_k(x)` instead of `k = self.c_k(x_k)`
   - V shift unchanged: `x_v = x * mv + x_prev * (1 - mv)`, `v = self.c_v(x_v)`

3. **`init_weights`**: Removed `time_mix_k.fill_(0.0)`.

Total params removed: 512 * 8 = 4,096. Net simplification.

## Why

- **Q shift failed** (exp127, +0.008): shifting projections that participate in attention scoring hurts
- **K is like Q** for scoring: `score = Q @ K^T`. K determines "what this token looks like as a key." Shifting K blurs the identity of "which position am I?"
- **RoPE applied after K projection**: shifted K means RoPE encodes the position of a mixed-position signal, degrading position encoding quality
- **V has no RoPE**: V determines "what information to aggregate." Shifting enriches values with neighbor context without interfering with position encoding
- **Simplification**: fewer params, less compute, cleaner design

## Risks

- **Neutral (medium)**: K shift might have been the main contributor, not V shift. Removing it eliminates the gain.
- **Regression (low)**: K+V shifts could have been synergistic. Removing K disrupts the balance.
- **Crash (none)**: Pure code removal, strictly simpler than exp126.
- Expected range: -0.002 to +0.002

## If it works

- Token shift is confirmed V-only. K shift was noise/harmful.
- Try V shift with different initialization (per-layer, warmer/cooler)
- Try V shift with wider lookback (mix in position -2 as well for trigram patterns)
- Focus on FP8 retry as the remaining "more steps" lever

## If it fails

- K shift was the beneficial component, or K+V synergy was needed
- Revert to exp126 (K+V shift)
- Try K-only shift as an alternative ablation
- Token shift tuning is largely done; focus effort elsewhere


## Résultats
- val_bpb: 1.074396
- peak_vram_mb: 11469.0
- status: discarded
