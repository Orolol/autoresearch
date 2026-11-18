Extend Value Embedding to ALL 8 layers with shared weights for universal token identity in value computation.

## What changed

Modified `has_ve()` to return `True` for all layers (was: only layer 0 and last layer). All 8 value_embeds share a single embedding table via weight tying. Each layer has its own independent ve_gate (input-dependent, 128 params each). Net change: +768 parameters (6 new ve_gates), ~2 lines modified in train.py.

Before: VE on layers 0 and 7 (shared weight, 2 gates).
After: VE on layers 0-7 (shared weight, 8 gates).

## Why

VE is the single most critical component in the model:
- Removing VE entirely: +0.023 catastrophe (exp1)
- Adding VE to layer 0 (exp84): -0.004 improvement (biggest single win in the series)

The pattern is clear: more VE coverage = better. Currently layers 1-6 compute V purely from projected representations with no direct token identity signal. By extending VE to all layers:
- Every layer's attention gets token-specific value residuals from step 0
- Gradient flow through the value path is improved at every layer, not just boundaries
- The input-dependent gate (`2 * sigmoid(ve_gate(x[:,:,:32]))`) lets each layer learn its own VE usage pattern — some may amplify VE (gate > 1), others may suppress it (gate < 1)
- Shared weights ensure the embedding table learns a single high-quality token representation, while gates handle layer-specific adaptation

This follows the dominant winning pattern: gradient flow improvements at near-zero compute cost.

## Risks

- Zero crash/OOM risk (same ops already used in layers 0 and 7, torch.compile compatible)
- Negligible speed impact (~6 extra embedding lookups on shared table, memory-bound, ~0.1% of budget)
- If mid-layer representations are rich enough without VE: neutral result
- If shared weights force a compromise across 8 diverse layer needs: mild regression ~0.001-0.002
- Worst case: ~0.003 regression (similar to other failed gradient flow changes)

## If it works

- Try separate (unshared) VE tables for different layer groups (early/mid/late)
- Try VE with larger gate (ve_gate_channels=64 instead of 32) for more expressive gating
- Try VE on subset of layers (0, 3, 4, 7 = boundaries + U-Net inflection) if full coverage is overkill

## If it fails

- Full coverage dilutes VE's effectiveness — first+last layer bookending is the sweet spot
- Try skip_lambdas init 0.05 (exp82 crashed on infra, never tested on GPU)
- Try DeltaNet or GLA (replace softmax attention entirely — bold architectural change)
- Try wider multi-scale RoPE range (1000, 5000, 50000, 500000) for more aggressive head specialization


## Résultats
- val_bpb: 1.081672
- peak_vram_mb: 11031.7
- status: discarded
