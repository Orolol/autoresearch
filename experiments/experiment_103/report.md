Sandwich norm (post-sublayer RMSNorm) on attention and MLP outputs to normalize residual stream contributions — 4th attempt after 3 infra crashes.

## What changed

Two-line change in `Block.forward` (line 132-133):

Before:
```python
x = x + self.attn(norm(x), ve, cos_sin, window_size)
x = x + self.mlp(norm(x))
```

After:
```python
x = x + norm(self.attn(norm(x), ve, cos_sin, window_size))
x = x + norm(self.mlp(norm(x)))
```

Each sublayer output is now wrapped in RMSNorm before being added to the residual stream. This creates a "sandwich" pattern: pre-norm (already existed) + post-norm (new).

## Why

1. Without post-norm, sublayer contributions accumulate and the residual stream grows ~O(sqrt(2*n_layer)) across 8 layers, forcing resid_lambdas/x0_lambdas to compensate for scale rather than controlling information flow.
2. Sandwich norm normalizes each sublayer contribution to O(1) scale, making the residual stream more stable.
3. Proven technique from PaLM, CogView, and other architectures.
4. Different from mid-MLP norm (exp79, +0.002, dead) which was INSIDE the MLP — sandwich norm is OUTSIDE, normalizing the full sublayer output.
5. exp100/101/102 all crashed on RunPod infra (CUDA init failure streak), never tested on GPU.

## Risks

- Very low. 2-line change using existing `norm()` function.
- c_proj zero-init means norm(0)=0 at init → model starts identically to baseline.
- Gradient amplification through norm at near-zero values mitigated by Muon's scale-invariant Newton-Schulz.
- Expected range: -0.003 to +0.005.

## If it works

- Try attention-only post-norm (SubLN) — remove MLP post-norm to check which matters more.
- Try learnable post-norm scale (gamma parameter) for finer control.

## If it fails

- Residual stream growth is not a bottleneck at DEPTH=8. Mark sandwich norm as dead.
- Focus on DiffAttn (exp99, also never tested on GPU) or DeltaNet/GLA.
