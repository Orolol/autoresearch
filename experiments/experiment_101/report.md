Retry sandwich norm (post-sublayer RMSNorm) — exp100 crashed on infra, never tested on GPU.

## What changed

Two lines modified in `Block.forward`:

```python
# Before (pre-norm only):
x = x + self.attn(norm(x), ve, cos_sin, window_size)
x = x + self.mlp(norm(x))

# After (sandwich norm = pre-norm + post-norm):
x = x + norm(self.attn(norm(x), ve, cos_sin, window_size))
x = x + norm(self.mlp(norm(x)))
```

Each sublayer output is RMS-normalized before being added to the residual stream, bounding each sublayer's contribution to O(1) scale.

## Why

- Exp100 attempted this exact change but crashed on RunPod infra (CUDA init failure pattern visible across exp99-210). The code was never tested on GPU.
- Without post-norm, the residual stream magnitude grows ~O(sqrt(2*n_layer)) as each sublayer adds unnormalized contributions. For 8 layers (16 sublayer adds), the residual can reach ~4x initial embedding scale.
- Sandwich norm (pre + post norm) is used in PaLM, CogView, and other large-scale architectures to stabilize deep residual streams.
- Different from mid-MLP norm (exp79, +0.002, dead): mid-MLP normalizes INSIDE the MLP. Sandwich norm normalizes OUTSIDE, controlling scale of contributions to the shared residual stream.
- Zero new parameters, zero new modules, minimal compute overhead (2 extra F.rms_norm per layer).
- Safe with c_proj zero-init: norm(0) = 0, so the model starts identically to baseline.

## Risks

- Gradient amplification through norm at near-zero values in early training. Mitigated by Muon's scale-invariant Newton-Schulz for matrix params and AdamW's second-moment normalization for non-matrix params.
- Mid-MLP norm was dead, but sandwich norm addresses a fundamentally different issue (residual stream growth vs. MLP internal scale).
- Expected range: -0.003 to +0.005
