Sandwich norm (post-sublayer RMSNorm) on attention and MLP outputs — 3rd attempt after 2 infra crashes.

## What changed

In `Block.forward`, wrapped both sublayer outputs with `norm()` before adding to the residual stream:

```python
# Before:
x = x + self.attn(norm(x), ve, cos_sin, window_size)
x = x + self.mlp(norm(x))

# After:
x = x + norm(self.attn(norm(x), ve, cos_sin, window_size))
x = x + norm(self.mlp(norm(x)))
```

## Why

- **Never tested on GPU**: exp100 and exp101 both crashed on RunPod infra (CUDA init failure), the code itself was never validated.
- **Residual stream growth**: Without post-norm, sublayer contributions accumulate and the residual stream grows ~O(sqrt(2*n_layer)). Normalizing each contribution to O(1) scale prevents this.
- **Different from mid-MLP norm**: exp79 (dead, +0.002) added norm INSIDE the MLP between activation and output projection. Sandwich norm is OUTSIDE, normalizing the full sublayer output before residual add.
- **Zero cost**: No new parameters, no new modules, minimal compute overhead (2 extra RMSNorm per layer).
- **Safe at init**: c_proj is zero-initialized, so norm(0)=0 at init — model starts identically to baseline.

## Expected outcome

Range: -0.003 to +0.005. The technique is well-established (PaLM, CogView) but may not help at DEPTH=8 where residual growth is modest.
