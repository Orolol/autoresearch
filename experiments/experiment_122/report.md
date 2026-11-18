Non-uniform MLP widths — layers 0-3 get 2x MLP expansion, layers 4-7 get 4x, iso-compute with current uniform 3x.

## What changed

Two changes in `train.py`:

1. **MLP.__init__** (line 112): Added `expansion` parameter (default=3). Hidden dim is now `expansion * config.n_embd` instead of hardcoded `3 * config.n_embd`.

2. **Block.__init__** (line 127): Computes `expansion = 2 if layer_idx < half else 4` where `half = config.n_layer // 2`. Layers 0-3 get 2x MLP (hidden=1024), layers 4-7 get 4x MLP (hidden=2048).

Total MLP params: 4 × (512×1024×2) + 4 × (512×2048×2) = 4,194K + 8,389K = 12,583K. Identical to 8 × (512×1536×2) = 12,583K (current uniform 3x). Zero net parameter change.

## Why

- **Later layers need more capacity**: In transformers, later layers handle increasingly abstract composition and fine-grained prediction. Wider MLPs give them more representational power.
- **U-Net synergy**: Skip connections (exp71) feed early-layer outputs to mirror late layers. Layers 4-7 receive combined information from the residual stream AND skip connections — wider MLPs give them more capacity to exploit this richer input.
- **Iso-compute**: Same total params and FLOPs as baseline. Any improvement is pure architectural benefit, not from more/fewer training steps.
- **MLP is high-leverage**: MLP 3x was the single biggest improvement (-0.008). MLP configuration directly impacts model quality.

## Risks

- Early layers with 2x MLP may become bottlenecks (MLP 2.5x uniform was +0.003)
- The wider late layers might not compensate for narrower early layers
- Zero crash/OOM risk
- Expected range: -0.005 to +0.005

## If it works

- Try 1.5x/4.5x split for even stronger asymmetry
- Try gradual increase (2x, 2.5x, 3x, 3.5x, 3.5x, 4x, 4x, 4.5x)
- Try assigning widths by VE presence (bookend layers 0,7 get 4x, others get 2.67x)

## If it fails

- Non-uniform MLP is dead at this scale; uniform 3x is optimal
- Try FP8 matmuls retry with more careful torch._scaled_mm implementation
- Try DeltaNet/GLA for fundamental attention replacement


## Résultats
- val_bpb: 1.068180
- peak_vram_mb: 10445.0
- status: IMPROVED
