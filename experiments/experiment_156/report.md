Partial Rotary Position Embedding (50%) — apply RoPE to only the first 64 of 128 head dimensions, leaving the rest position-free for content-only attention matching.

## What changed

In hyperparameters section (train.py line 549):
- Added `ROPE_FRACTION = 0.5` — fraction of head_dim that gets rotary encoding

In `apply_rotary_emb` (train.py line 52-61):
- Changed from applying rotation to all dims, to splitting x into `x_rope` (first `rope_dim` dims) and `x_pass` (remaining dims)
- Only rotates `x_rope`, then concatenates `[y1, y2, x_pass]`
- `rope_dim` is derived from `cos.shape[-1] * 2`, so it adapts automatically

In `GPT.__init__` (train.py line 216):
- Added `self.rope_dim = max(2, int(head_dim * ROPE_FRACTION) // 2 * 2)` — ensures even dim
- Passes `self.rope_dim` (64) instead of `head_dim` (128) to `_precompute_rotary_embeddings`

In `GPT.init_weights` (train.py line 253):
- Uses `self.rope_dim` instead of `head_dim` for rotary recomputation

Total new parameters: 0. Compute change: slightly less (fewer RoPE multiplications, smaller cos/sin buffers).

## Why

QK-norm equalizes magnitudes, and RoPE encodes position in ALL 128 dims. Every attention score is position-dependent. The model has no way to say "these two tokens are similar regardless of their positions."

Partial RoPE splits each head's Q*K dot product into two additive components:
- **64 rotated dims**: position-sensitive similarity (with per-head multi-scale RoPE base)
- **64 non-rotated dims**: pure content similarity (position-independent)

This is a proven technique in production models:
- PaLM: 25% rotary dims
- CodeLlama: partial RoPE for extended context
- Gemma: partial RoPE

With multi-scale RoPE (bases 2500, 10000, 40000, 160000), each head already specializes by distance scale. Partial RoPE adds an orthogonal axis: content-only matching. A head with base=2500 now does both "nearby position" matching (rotated part) AND "similar content" matching (non-rotated part).

The Q/K linear projections are per-head, so even the non-rotated dims produce head-specific content similarity. Heads can learn different content-matching patterns in their non-rotated dims.

## Risks

- **Regression (medium)**: 50% is aggressive. At 2048 context, the model may need more positional dims. But the non-rotated dims still learn meaningful similarity from Q/K projections — they just don't encode position.
- **Neutral (medium)**: With only 4 heads and already-diverse multi-scale RoPE, content-only matching may not provide enough benefit.
- **Crash (none)**: Standard tensor slicing and cat. All shapes static. torch.compile compatible.
- **OOM (none)**: Reduces cos/sin buffer size by 50%.

Expected range: -0.003 to +0.005.

## If it works

- Try ROPE_FRACTION=0.25 (PaLM-style, only 32 rotated dims)
- Try ROPE_FRACTION=0.75 (more conservative, 96 rotated dims)
- Try per-head ROPE_FRACTION (e.g., short-range heads get more rotary, long-range heads get more content)
- Try learnable rope_fraction (input-dependent partial rotation)

## If it fails

- Position encoding is already well-optimized (multi-scale RoPE tuning done)
- The model benefits more from full positional encoding at 2048 context
- Move to FP8 matmuls (biggest "more steps" lever) or DeltaNet/GLA
