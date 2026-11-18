Replace ReluSquared with GeluSquared activation in MLP for softer gradient flow through negative activations.

## What changed

Single line change in `MLP.forward`: `F.relu(x).square()` -> `F.gelu(x).square()` (line 119 of train.py).

## Why

ReluSquared has a hard dead zone: for any input x <= 0, the output and gradient are exactly 0. In the severely undertrained regime (~830 steps, 5 min), this means roughly half of all MLP activations produce zero gradient per forward pass, wasting learning capacity. Neurons that happen to receive negative inputs from the current batch learn nothing.

GeluSquared replaces the hard ReLU gate with GELU's soft gate:
- For x >> 0: GELU(x) ≈ x, so GELU(x)² ≈ x² — identical to ReluSquared
- For x << 0: GELU(x) ≈ 0, so GELU(x)² ≈ 0 — identical to ReluSquared
- For x near 0: GELU(x) ≈ 0.5x, so GELU(x)² ≈ 0.25x² — smooth transition instead of hard cutoff

This smooth transition means:
1. Gradients flow through mildly negative activations (neurons can "recover")
2. The loss landscape is smoother (continuous second derivative at x=0)
3. The square still provides sparsity (small activations → even smaller)

Note: SwiGLU was tested (exp30, 1.083) but it changed the MLP *structure* (added gating path, reduced hidden dim for iso-compute). This change only swaps the activation function, keeping identical architecture and parameters.

GELU is the standard activation in modern LLMs (GPT-2, BERT, LLaMA). The combination GELU + squared is novel but well-motivated.

## Initialization compatibility

c_proj is initialized to zeros, so the MLP output is zero at step 0 regardless of activation. The model starts from exactly the same point as the baseline.

## Risks

- Zero crash/OOM risk (no new parameters, no structural changes)
- Zero speed impact (GELU is marginally slower than ReLU, negligible vs matmul cost)
- If exact sparsity of ReLU is important for this architecture: regression ~0.001-0.005
- If soft gradients help optimization: improvement ~0.0005-0.002

## If it works

- Try other squared activations: Mish², Swish²
- Combine with other promising changes (z-loss tuning, momentum warmup)

## If it fails

- Confirms ReluSquared's hard sparsity is beneficial at this scale
- Try GeGLU (GELU-gated linear unit) at iso-compute
- Move to more radical architectural changes: Differential Transformer, simple MoE


## Résultats
- val_bpb: 1.084504
- peak_vram_mb: 10337.7
- status: discarded
