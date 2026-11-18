Differential Attention (DiffTransformer) — split each head into 2 sub-heads, subtract paired attention outputs to cancel noise, and halve c_proj input dimension.

## What changed

Two changes in `CausalSelfAttention`:

1. **c_proj dimension** (`__init__`): `nn.Linear(n_embd, n_embd)` changed to `nn.Linear(n_embd // 2, n_embd)`. Reduces c_proj params from 512x512=262K to 512x256=131K per layer (8 layers: ~1M fewer params total).

2. **Differential attention** (`forward`): After QK-norm and before SDPA, reshape Q/K/V from `(B, T, n_head, HEAD_DIM)` to `(B, T, 2*n_head, HEAD_DIM//2)` using `unflatten(-1, (2, d2)).flatten(2, 3)`. Run SDPA with 8 sub-heads of dim 64 instead of 4 heads of dim 128 (same total FLOPs). After SDPA, reshape to `(B, T, n_head, 2, d2)` and subtract: `y = y[:,:,:,0,:] - 0.5 * y[:,:,:,1,:]`. Result: `(B, T, n_embd//2)` fed into the smaller c_proj.

The split happens AFTER RoPE application, so sub-head 0 captures low-frequency positional patterns and sub-head 1 captures high-frequency. The subtraction cancels attention patterns common to both frequency bands.

## Why

- DiffTransformer (Microsoft, 2024) shows noise cancellation via attention subtraction improves quality, especially for information retrieval and long-context tasks
- The c_proj reduction saves ~5% of per-layer linear compute, giving slightly more training steps in the 300s budget
- No new parameters, no learnable lambda (fixed at 0.5), fully static shapes → zero torch.compile risk
- All existing components (VE, multi-scale RoPE, U-Net skips) are unaffected — VE is applied before the split, RoPE is applied before the split

## Risks

- HEAD_DIM//2=64 sub-heads produce softer attention (QK-norm at 128 dims means each 64-dim half has norm ~1/sqrt(2))
- Halving value dimension per logical head from 128 to 64 reduces information capacity
- Fixed lambda=0.5 may not be optimal (paper uses learnable ~0.8)
- HEAD_DIM=64 as standalone experiment was +0.012, but DiffAttn is fundamentally different (subtraction, not just smaller heads)
- Expected range: -0.005 to +0.015

## If it works

- Try learnable lambda (per-head, per-layer) instead of fixed 0.5
- Try lambda=0.8 (paper's preferred value)
- Try applying DiffAttn to only certain layers (e.g., deep layers only)

## If it fails

- The information capacity loss from 128→64 value dim outweighs noise cancellation
- Try DiffAttn with doubled heads (8 logical heads, 16 sub-heads) to maintain output dimension without c_proj change
- Focus on remaining ideas: Schedule-Free Adam, FP8 matmuls, wider RoPE range
