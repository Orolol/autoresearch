Gradual MLP width ramp — linear expansion [1,2,2,3,3,4,4,5] across 8 layers, iso-compute with previous 2x/4x split.

## What changed

Replaced the 2-step MLP width split (layers 0-3: 2x, layers 4-7: 4x) with a smooth linear ramp from 1x to 5x:
- `expansion = round(1 + 4 * layer_idx / (n_layer - 1))`
- Layer expansions: [1, 2, 2, 3, 3, 4, 4, 5], sum=24, average=3x
- Hidden dims: [512, 1024, 1024, 1536, 1536, 2048, 2048, 2560]
- Total MLP params/FLOPs identical to baseline (iso-compute)

## Why

Exp122 proved non-uniform MLP widths work (-0.00114 with 2x/4x). This tests a more aggressive and gradual version: layer 7 gets 5x (vs 4x before) while layer 0 shrinks to 1x (vs 2x). The hypothesis is that later layers benefit from even more capacity for fine-grained prediction, while early layers mainly do feature mixing and can tolerate less MLP width.

## Risks

- Layer 0 at 1x MLP (hidden=512) may be a bottleneck
- MLP 2.5x uniform was +0.003, suggesting narrow MLPs can hurt — but here only 1 layer is at 1x
- If this regresses: try intermediate [2,2,2,3,3,4,4,4] (gentler ramp, minimum 2x)
