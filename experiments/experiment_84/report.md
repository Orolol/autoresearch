Extend Value Embedding to first AND last layer with shared weights for bookend token identity in value computation (retry of exp83 which crashed on infra).

## What changed

Modified `has_ve()` to return True for layer 0 in addition to last layer. Added weight tying so both layers share the same embedding table. Each layer has its own independent VE gate (input-dependent, 128 params each). ~5 lines changed in train.py.

Before: VE on layer 7 only. After: VE on layers 0 and 7, shared embedding weights, separate gates.

## Why

VE is the most critical component in this model (removing it = +0.023 catastrophe in exp1). Currently only the last layer benefits from direct token identity in its value computation. The first layer computes V purely from projected input embeddings with no token-specific residual.

By adding shared VE to layer 0:
- The first layer's attention immediately gets token-specific value information from step 0
- Gradient flow through the value path is improved at both ends of the network (bookending)
- The input-dependent gate (`2 * sigmoid(ve_gate(x[:,:,:32]))`) ensures each layer applies VE differently despite shared weights
- Zero additional parameters (shared embedding), one extra ve_gate (128 params)
- Negligible compute cost (one extra embedding lookup per batch)

This follows the pattern of all winning experiments: gradient flow improvements (softcap 13, z-loss, U-Net skips, multi-scale RoPE) at minimal compute cost.

Exp83 tested this exact idea but crashed on infra (never ran on GPU). This is a retry.

## Risks

- Zero crash/OOM risk (weight sharing is standard PyTorch, ve_gate uses same ops as existing one)
- Zero speed impact (one extra embedding lookup is memory-bound and tiny vs matmuls)
- If first-layer representations are too primitive to benefit from VE: neutral result
- If shared weights force a compromise between first and last layer VE needs: mild regression ~0.001
- Worst case: ~0.002-0.003 regression

## If it works

- Try VE on all 8 layers with shared weights (every layer gets token identity)
- Try separate (unshared) VE for first and last layer for independent specialization
- Try VE on first + middle (layer 4, U-Net inflection) + last layer
- Widen multi-scale RoPE range (1000, 5000, 50000, 500000)

## If it fails

- First layer is too early for VE to help — token representations haven't been processed yet
- Try skip_lambdas init 0.05 (retry of exp82 which also crashed on infra)
- Try DeltaNet or GLA (replace softmax attention entirely — bold architectural change)
- Try wider multi-scale RoPE range for more aggressive head specialization


## Résultats
- val_bpb: 1.072876
- peak_vram_mb: 10450.8
- status: IMPROVED
