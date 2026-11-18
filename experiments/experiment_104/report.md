Sandwich norm (post-sublayer RMSNorm) on attention and MLP outputs — 5th attempt after 4 infra crashes.

## What changed

In `Block.forward`, wrapped both attention and MLP sublayer outputs with `norm()` before adding to the residual stream:

```python
# Before (baseline):
x = x + self.attn(norm(x), ve, cos_sin, window_size)
x = x + self.mlp(norm(x))

# After (sandwich norm):
x = x + norm(self.attn(norm(x), ve, cos_sin, window_size))
x = x + norm(self.mlp(norm(x)))
```

## Why

Without post-norm, each sublayer's output is added raw to the residual stream. Over 8 layers (16 sublayer additions), the residual stream magnitude grows ~O(sqrt(2*n_layer)), which can destabilize later layers and force the learned resid_lambdas/x0_lambdas to compensate for scale drift instead of controlling information flow.

Sandwich norm (pre+post normalization) normalizes each sublayer's contribution to O(1) scale, keeping the residual stream magnitude bounded. This is a proven technique used in PaLM, CogView, and other large-scale architectures.

This is distinct from mid-MLP norm (exp79, +0.002, dead), which was INSIDE the MLP between activation and output projection. Sandwich norm is OUTSIDE, normalizing the full sublayer output.

## Properties

- Zero new parameters, zero new modules
- Minimal compute overhead (2 extra RMSNorm per layer)
- c_proj zero-init means norm(0)=0 at init, so model starts identically to baseline
- Muon's Newton-Schulz is scale-invariant, so gradient amplification through norm at near-zero values is harmless
- torch.compile safe (uses existing F.rms_norm)

## Previous attempts

- exp100: infra crash (CUDA init failure)
- exp101: infra crash
- exp102: infra crash
- exp103: infra crash
- exp104: this attempt (same code, hoping infra is stable now)
