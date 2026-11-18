Reduce local attention block_size from 256 to 128 for middle layers (2-5) to gain more training steps.

## What changed

In `CausalSelfAttention.__init__` (train.py line 85):
- `self.block_size = 256` -> `self.block_size = 128`

This affects layers 2-5 (where `self.local_attn = True`). All other layers (0, 1, 6, 7) still use full causal attention.

Zero new parameters. Zero new memory. Pure compute reduction.

## Why

Exp149 proved block-diagonal local attention works (-0.001546, new best 1.063080). Exp150 confirmed layer selection is done (extending to 1-6 was neutral). The documented next step (prompt.md item #44) is block_size tuning.

block_size=128 halves local attention compute per layer compared to 256:
- Each position attends to 128 tokens instead of 256
- T=2048 / 128 = 16 blocks (exact division, all shapes static)
- 4 local layers × 50% cheaper = ~6-7% total compute reduction → more training steps

The "more steps" lever has been the most consistent win historically:
- MLP 3x (exp14, -0.008): biggest single win ever
- Block-diagonal local attention (exp149, -0.001546): current best

Boundary effects are mitigated by:
- Token shift: 1-token cross-boundary lookback for K/V
- Full-attention layers 0, 1, 6, 7 provide global context
- U-Net skip connections propagate global features
- x0_lambdas: direct access to initial embedding

## Risks

- **Regression (medium)**: 128 tokens is quite narrow. Harder boundaries limit local context. Some layers may struggle without 256-token context windows.
- **Neutral (medium)**: If MLP dominates compute (especially in late layers with 4.5x expansion), the additional attention savings may not yield enough extra steps to matter.
- **Crash**: None — identical code path to exp149, T=2048 divides evenly by 128.

Expected range: -0.003 to +0.005.

## If it works

- Try block_size=96 or 64 to push further
- Try non-uniform block sizes: 128 for early local layers (2-3), 256 for late local layers (4-5)

## If it fails

- block_size=256 is confirmed optimal. Block_size tuning done.
- Try block_size=512 as the softer-boundary alternative
- Move to FP8 matmuls (biggest "more steps" lever) or DeltaNet/GLA


## Résultats
- val_bpb: 1.063407
- peak_vram_mb: 12499.2
- status: discarded
