Increase DEPTH from 8 to 9 layers (ASPECT_RATIO=56) for deeper modeling capacity with accumulated gradient flow improvements.

## What changed

Two constant changes in train.py:
- `DEPTH = 9` (was 8)
- `ASPECT_RATIO = 56` (was 64, to keep model_dim=512 after rounding)

Result: 9 transformer layers instead of 8, all at 512-dim with 4 heads. Everything else adapts automatically:
- U-Net skips: 4 early layers (0-3) save, 4 late layers (5-8) receive, layer 4 is middle
- VE on layers 0 and 8 (shared weights, separate gates)
- Multi-scale RoPE: 4 bases for 4 heads, unchanged
- skip_lambdas: 4 skip connections (same count, but spanning 9 layers)

## Why

Depth is the single most valuable architectural axis:
- exp18 (DEPTH=7): +0.033 catastrophe (losing 1 layer was devastating)
- exp22 (DEPTH=9 iso-compute): +0.001 from baseline (marginal regression)

But exp22 was tested BEFORE multi-scale RoPE (exp80: -0.0002), U-Net skips (exp71: -0.001), and VE first+last (exp84: -0.004). These gradient flow improvements specifically help deeper models by:
1. Providing shorter gradient paths through U-Net skips (now spanning 9 layers)
2. Giving token identity signals at both ends of a deeper network via VE
3. Specializing attention heads across a wider frequency range via multi-scale RoPE

Also, exp22 used MLP 2.5x (known to be -0.0026 worse than 3x per exp15). This experiment uses MLP 3x.

The compute overhead is ~8-10% (not 12.5% because lm_head and embeddings are fixed costs), meaning ~750 steps instead of ~830. The extra transformer block provides a full set of attention + MLP operations worth of additional capacity.

## Risks

- Zero crash/OOM risk (standard Block addition, no new code)
- ~8-10% fewer training steps (~750 vs ~830) due to extra layer compute
- If step reduction outweighs capacity gain: mild regression ~0.001-0.003
- If 9 layers need more training time to converge properly: could regress ~0.002
- Worst case similar to exp22: +0.001 regression (but with better gradient flow, could break even or improve)

## If it works

- Try DEPTH=10 (push depth even further with the improved gradient flow)
- Try reducing MLP from 3x to 2.75x to get more steps with 9 layers
- Try wider RoPE base range (1000, 5000, 50000, 500000) for more aggressive head specialization across 9 layers
- Try skip_lambdas init 0.05 (exp82 crashed on infra, never tested) for faster U-Net activation

## If it fails

- 8 layers is the optimal depth for 300s training budget even with improved gradient flow
- Try skip_lambdas init 0.05 (untested, crashed on infra in exp82)
- Try wider multi-scale RoPE range (1000, 5000, 50000, 500000)
- Try DeltaNet or GLA (replace softmax attention entirely)
- Try per-sublayer output scaling (attn_alpha, mlp_alpha per layer)
