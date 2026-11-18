Reduce RoPE base frequency from 10000 to 1000 for better positional resolution within 2048 context length.

## What changed

Single constant change: `_precompute_rotary_embeddings` default parameter `base=10000` changed to `base=1000`.

## Why

With base=10000 and head_dim=128 (64 rotary dimensions), the wavelengths range from ~6.3 to ~54,600. Only ~21 out of 64 dimensions have wavelengths within the 2048 context length. The remaining 43 dimensions have wavelengths far exceeding the context — they barely rotate across the full sequence and provide almost no positional information. They're essentially constant-valued channels.

With base=1000, the maximum wavelength drops to ~5,672, and ~27+ dimensions become fully utilized within the 2048 context. This gives the model significantly more positional resolution — more RoPE dimensions that can meaningfully distinguish positions.

The model uses QK-norm after applying RoPE, so the rotary angles affect only the direction of Q/K vectors (attention pattern selectivity), not their magnitude. This makes the change safe.

base=10000 was the original RoPE default (designed for ~2048 context as a reasonable choice, not an optimized one). LLaMA 3 uses base=500000 for 8192+ contexts — the trend is to adjust base for context length. For our SHORT 2048 context, reducing the base is the natural direction.

## Risks

- Low crash/OOM risk: identical computation, same shapes, just different frequency values in a precomputed buffer
- If model benefits from high-wavelength dims as position-invariant channels: mild regression ~0.001-0.005
- If base=1000 is too low (positions too tightly resolved, reducing diversity in attention patterns): try base=2000 or 5000 next

## If it works

- Try base=500 or 200 for even more resolution
- Try per-layer RoPE base (lower base in early layers for local patterns, higher in late layers for global)
- Combine with softcap=12 (never successfully tested — crashed 5x on infra)

## If it fails

- Try base=5000 (more conservative reduction)
- Move to other architectural changes: GeGLU, simple MoE, Differential Transformer
- Consider learnable per-head attention temperature as alternative positional encoding improvement


## Résultats
- val_bpb: 1.091849
- peak_vram_mb: 10337.7
- status: discarded
