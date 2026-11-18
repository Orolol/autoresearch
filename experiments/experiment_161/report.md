Layer-conditional activation — plain ReLU for early layers (0-3), ReluSquared for late layers (4-7).

## What changed

In `MLP.__init__` (train.py):
- Added `use_squared` parameter (default True for backward compatibility)
- Stored as `self.use_squared` instance attribute

In `MLP.forward` (train.py):
- Activation is now conditional: `F.relu(x).square()` if `use_squared` else `F.relu(x)`
- `use_squared` is a constant per instance, so torch.compile traces it as a static branch

In `Block.__init__` (train.py):
- Early layers (0-3, layer_idx < half): `use_squared=False` (plain ReLU)
- Late layers (4-7, layer_idx >= half): `use_squared=True` (ReluSquared, unchanged)

Total new parameters: 0. Compute change: negligible savings (no square op in 4 early layers).

## Why

Early layers have narrow MLPs (1.5x expansion = 768 hidden units). ReluSquared applies two filters:
1. ReLU: kills ~50% of neurons (negative values)
2. Squaring: suppresses small positive activations toward zero

With only 768 hidden units and double filtering, the effective active capacity of early MLPs is severely limited. Plain ReLU preserves all positive activations at full magnitude, giving early layers more "active neurons" to work with.

Late layers have wide MLPs (4.5x = 2304 hidden units). They have plenty of capacity. ReluSquared's sparsity is BENEFICIAL here — it forces the wide MLP to specialize and avoids diffuse activations.

This mirrors the non-uniform MLP width philosophy (exp122/129): early layers get different treatment than late layers. Width is already non-uniform (1.5x vs 4.5x); this extends the differentiation to the activation function.

c_proj is zero-initialized (line 231), so at the start of training MLP output is exactly 0 regardless of activation. The magnitude difference between ReLU and ReluSquared is compensated by c_proj's learned scaling.

**Different from dead activation experiments:**
- Exp66 (GeluSquared): changed ALL layers to GeluSquared → +0.006
- Exp30 (SwiGLU): changed ALL layers to SwiGLU → +0.004
- Both changed activation UNIFORMLY. This experiment is LAYER-CONDITIONAL, matching activation to layer role.

## Risks

- **Neutral (high)**: The model may converge to similar behavior regardless — c_proj compensates for activation differences, and the optimization landscape may be similar.
- **Regression (medium)**: ReluSquared's sparsity might actually help narrow early MLPs by forcing selectivity. Without squaring, early MLPs might waste capacity on low-information features.
- **Crash (none)**: `F.relu(x)` is standard. The `if self.use_squared` branch is a constant per MLP instance — torch.compile traces it as a static branch, no dynamic control flow.
- **OOM (none)**: Slightly less memory (no intermediate squared tensor in 4 early layers).

Expected range: -0.002 to +0.003.

## If it works

- Try the opposite: ReLU for late layers, ReluSquared for early (test the hypothesis direction)
- Try a gradient: layers 0,1 ReLU, 2,3 ReLU, 4,5 ReluSquared, 6,7 Relu^3 (increasing non-linearity)
- Try ReLU^1.5 for early layers (compromise between ReLU and ReluSquared)
- Apply the same layer-conditional logic to other hyperparameters (e.g., different token shift init per layer)

## If it fails

- ReluSquared is optimal regardless of MLP width — the squaring's benefit (sparsity, gradient scaling) applies equally at all widths
- Activation function axis is truly dead (uniform ReluSquared is the global optimum)
- Next directions: FP8 matmuls (biggest "more steps" lever), DeltaNet/GLA (full attention replacement)
- Consider that 1.062436 may be near the architecture's practical ceiling at ~830 steps


## Résultats
- val_bpb: 1.066087
- peak_vram_mb: 12310.5
- status: discarded
