Extend local attention from layers 2-5 to layers 1-6 (bookend-only full attention), aligning VE layers with global context.

## What changed

In `CausalSelfAttention.__init__`, changed the local attention condition from `layer_idx >= 2 and layer_idx < n - 2` to `not has_ve(layer_idx, n_layer)`. This means only layers 0 and 7 (the VE bookend layers) keep full causal attention, while all other layers (1-6) use block-diagonal local attention with block_size=256.

## Why

Exp149 proved block-diagonal local attention works (-0.001546, new best 1.063080). The prompt for exp149 noted: "If works: tune block_size (128, 512), vary which layers are local." Extending to 6/8 local layers (up from 4/8) gives ~50% more compute savings from attention, translating to more training steps — the same lever as the MLP 3x win (exp14, -0.008).

The architectural rationale is clean: VE layers need full sequence context to leverage token identity for all positions, while non-VE layers do local refinement. Layers 1 and 6 still get global information through:
- Residual stream from layer 0 (full attention)
- U-Net skip connections (layer 0→7, 1→6, 2→5, 3→4)
- Token shift (1-token cross-boundary lookback)
- x0_lambdas (direct access to initial embedding)

## Risk assessment

- **Regression** (medium): Layers 1 and 6 lose global attention. Layer 6 in particular is close to output and may benefit from global context. But it receives skip connection from layer 1 and has token shift.
- **Neutral** (medium): Attention may not be the compute bottleneck in these layers (MLP is larger in late layers due to 4.5x expansion).
- **Crash**: None — identical code path to exp149.

Expected range: -0.003 to +0.005.


## Résultats
- val_bpb: 1.063298
- peak_vram_mb: 12499.2
- status: discarded
