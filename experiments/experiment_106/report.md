Sandwich norm (post-sublayer RMSNorm) on attention and MLP outputs — 6th attempt after 5 infra crashes.

## What changed

Two-line change in `Block.forward` (lines 131-133):

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

Each sublayer output is now wrapped in RMSNorm before being added to the residual stream, creating a "sandwich" pattern (pre-norm + post-norm).

## Why

1. **Never tested on GPU**: exp100-104 all crashed on RunPod infra (CUDA init failures). Exp105 ran successfully, confirming infra is now stable.
2. **Residual stream growth**: Without post-norm, sublayer contributions accumulate and the residual stream grows ~O(sqrt(2*n_layer)) across 8 layers (16 sublayer additions). Normalizing each contribution to O(1) scale keeps the residual stream bounded.
3. **Proven technique**: Used in PaLM, CogView, and SubLN architectures.
4. **Different from mid-MLP norm**: exp79 (+0.002, dead) added norm INSIDE the MLP between activation and output projection. Sandwich norm is OUTSIDE, normalizing the full sublayer output before residual add.
5. **Zero cost**: No new parameters, no new modules, minimal compute overhead (2 extra RMSNorm per layer).
6. **Safe at init**: c_proj is zero-initialized, so norm(0)=0 at init — model starts identically to baseline.

## Risks

- Very low. 2-line change using existing `norm()` function.
- Normalizing sublayer outputs may remove useful magnitude information from the residual stream.
- Muon's Newton-Schulz is scale-invariant, so gradient amplification through norm at near-zero values is harmless for matrix params.
- Expected range: -0.003 to +0.005.

## If it works

- Try attention-only post-norm (SubLN) to check which sublayer benefits more.
- Try learnable post-norm scale (gamma parameter).
- Combine with other untested ideas (DiffAttn without V-halving, Schedule-Free Adam).

## If it fails

- Residual stream growth is not a bottleneck at DEPTH=8. Mark sandwich norm as dead.
- Move to Schedule-Free Adam, FP8 matmuls, or DeltaNet/GLA.


## Résultats
- val_bpb: 1.078373
- peak_vram_mb: 11478.3
- status: discarded
