Partial Rotary Position Embedding (50%) — retry of exp156 (infra crash), apply RoPE to only 64/128 head dims for content-only matching.

## What changed

In hyperparameters (train.py):
- Added `ROPE_FRACTION = 0.5` — fraction of head_dim that gets rotary encoding

In `apply_rotary_emb` (train.py):
- Changed `d = x.shape[3] // 2` to `d = cos.shape[-1]` — derives rope_dim from cos/sin shape
- Split x into `x[..., :d]`, `x[..., d:2*d]` (rotated), and `x[..., 2*d:]` (pass-through)
- Concatenates `[y1, y2, x_pass]` — when ROPE_FRACTION=1.0, x_pass is empty (backward compatible)

In `GPT.__init__` (train.py):
- Compute `self.rope_dim = max(2, int(head_dim * ROPE_FRACTION) // 2 * 2)` = 64 (ensures even)
- Pass `self.rope_dim` instead of `head_dim` to `_precompute_rotary_embeddings`

In `GPT.init_weights` (train.py):
- Use `self.rope_dim` for rotary recomputation

Total new parameters: 0. Compute change: slightly less (smaller cos/sin buffers, fewer rotary multiplications).

## Why

RoPE encodes position in ALL 128 Q/K dims. After QK-norm, every attention score is fully position-dependent. The model has no mechanism for "these tokens are similar regardless of position."

Partial RoPE splits each head's Q*K dot product into two additive components:
- **64 rotated dims**: position-sensitive similarity (with per-head multi-scale RoPE base)
- **64 non-rotated dims**: pure content similarity (position-independent)

This is a proven technique: PaLM uses 25% rotary, CodeLlama and Gemma use partial RoPE.

With multi-scale RoPE (bases 2500, 10000, 40000, 160000), each head already specializes by distance scale. Partial RoPE adds an orthogonal axis: content-only matching. The Q/K projections are per-head, so non-rotated dims learn head-specific content patterns.

Exp156 attempted this exact change but crashed with 0 VRAM and 0 val_bpb — characteristic of infra crash (RunPod), never reached GPU. This is a clean retry.

## Risks

- **Regression (medium)**: 50% is aggressive at 2048 context. Every dim may benefit from positional encoding.
- **Neutral (medium)**: 4 heads with already-diverse multi-scale RoPE may not gain from content-only matching.
- **Crash (very low)**: Standard tensor slicing and cat. All shapes static. torch.compile compatible. Exp156 crash was infra.
- **OOM (none)**: Reduces cos/sin buffer by 50%.

Expected range: -0.003 to +0.005.

## If it works

- Try ROPE_FRACTION=0.75 (conservative, 96 rotated dims)
- Try ROPE_FRACTION=0.25 (PaLM-style, 32 rotated dims)
- Try per-head ROPE_FRACTION (short-range heads more rotary, long-range heads more content)

## If it fails

- Full positional encoding needed at 2048 context with only 4 heads
- RoPE axis exhausted (multi-scale tuning + partial fraction both tested)
- Move to FP8 matmuls (biggest "more steps" lever) or DeltaNet/GLA


## Résultats
- val_bpb: 1.064943
- peak_vram_mb: 12492.3
- status: discarded
