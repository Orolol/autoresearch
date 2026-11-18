Sandwich norm (post-sublayer RMSNorm) on both attention and MLP outputs to normalize residual stream contributions.

## What changed

One change in `Block.forward` (2 lines modified):

```python
# Before (pre-norm only):
x = x + self.attn(norm(x), ve, cos_sin, window_size)
x = x + self.mlp(norm(x))

# After (sandwich norm = pre-norm + post-norm):
x = x + norm(self.attn(norm(x), ve, cos_sin, window_size))
x = x + norm(self.mlp(norm(x)))
```

Each sublayer output is RMS-normalized before being added to the residual stream. This bounds each sublayer's contribution to O(1) scale regardless of what the projection weights learned.

## Why

- Without post-norm, the residual stream magnitude grows ~O(sqrt(2*n_layer)) as each sublayer adds unnormalized contributions. For 8 layers (16 sublayer adds), the residual can reach ~4x the initial embedding scale, making later layers operate on increasingly different scales.
- Sandwich norm (pre + post norm) is used in PaLM, CogView, and other large-scale architectures specifically to stabilize deep residual streams.
- Different from mid-MLP norm (exp79, +0.002, dead): mid-MLP normalizes INSIDE the MLP (between activation and output projection), affecting c_proj input scale. Sandwich norm normalizes OUTSIDE, controlling the scale of contributions to the shared residual stream.
- Zero new parameters, zero new modules, minimal compute overhead (2 extra F.rms_norm calls per layer, each a single fused kernel).
- Safe with c_proj zero-init: norm(0) = 0, so the model starts identically to baseline.

## Risks

- Gradient amplification through norm at near-zero values (~1/sqrt(eps)) in early training. Mitigated by Muon's scale-invariant Newton-Schulz orthogonalization for all matrix params.
- Mid-MLP norm was dead, so "more norms" might not help at this scale. But the mechanisms are fundamentally different.
- Expected range: -0.003 to +0.005

## If it works

- Try sandwich norm on attention only (SubLN style from DiffTransformer paper) to isolate which sublayer benefits
- Try combining with DiffAttn (exp99, never ran due to infra crash)

## If it fails

- The residual stream growth is not a bottleneck at DEPTH=8
- Focus on bolder changes: retry DiffAttn, try DeltaNet/GLA, or Schedule-Free Adam
