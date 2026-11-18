Focal loss (gamma=0.5) to focus learning on hard-to-predict tokens instead of uniform cross-entropy weighting.

## What changed

Modified the training loss computation in `GPT.forward()` (train.py lines 358-373). When `reduction='mean'` (training), instead of uniform cross-entropy:

1. Compute per-token cross-entropy losses (`reduction='none'`)
2. Derive target probability: `p_target = exp(-per_token_loss)`
3. Compute focal weight: `(1 - p_target)^0.5` (gamma=0.5)
4. Normalize weights by their mean to preserve loss magnitude
5. Apply weighted mean

The eval path (`reduction='none'`) is completely unchanged.

## Why

- **Novel direction**: Loss function modifications have been barely explored. Only label smoothing (exp34, +0.040 catastrophic) and z-loss (exp9, kept) have been tried. Focal loss is fundamentally different from both.
- **Theoretical justification**: BPB = sum(nats) / sum(bytes). Hard tokens contribute disproportionately to total nats. A token with loss=5 contributes 10x more to BPB than a token with loss=0.5. Focusing gradient on hard tokens should reduce total nats more efficiently.
- **Different from label smoothing**: Label smoothing changes the TARGET distribution (one-hot → smoothed), telling the model "every token is possible." Focal loss keeps targets one-hot but changes the WEIGHT of each example. The model still learns to be confident on correct tokens.
- **Exhausted other directions**: 8+ consecutive neutral architectural experiments. Optimizer perfectly tuned. Hyperparameters at their optima. Loss function is the last unexplored axis.

## Risks

- **Regression (medium)**: Loss modifications are dangerous territory — label smoothing was +0.040. But focal loss has a completely different mechanism (reweighting vs target smoothing).
- **Neutral (high)**: gamma=0.5 is mild. Hard tokens get ~1.4x weight, easy tokens ~0.4x. May be too subtle to detect over 830 steps.
- **Gradient variance**: Per-token weights add noise to gradient estimates. Mitigated by weight normalization.
- **Crash (none)**: Pure loss computation change, no new parameters, no shape changes.
- Expected range: -0.003 to +0.010

## If it works

- Try stronger gamma (1.0, 2.0) for more aggressive hard-token focus
- Try adaptive gamma that increases during warmdown (start uniform, end focused)
- Combine with per-position weighting (later positions have more context → weight them more)

## If it fails

- Loss function axis is dead. The model's training objective is already optimal.
- Remaining high-potential ideas: FP8 matmuls (retry with compile-friendly approach), DeltaNet/GLA (complex but untested architecture)
- Consider that 1.067 may be the fundamental capacity limit for this architecture/compute budget


## Résultats
- val_bpb: 1.069145
- peak_vram_mb: 11984.2
- status: discarded
