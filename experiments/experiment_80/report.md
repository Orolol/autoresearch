Multi-scale RoPE: per-head base frequencies (2500, 10000, 40000, 160000) for attention head specialization across distance ranges.

## What changed
Added per-head RoPE base frequencies in geometric progression (ratio=4). Each of the 4 attention heads gets a different base:
- Head 0: base=2500 (local patterns, fast positional decay)
- Head 1: base=10000 (default, unchanged)
- Head 2: base=40000 (medium-range, slow decay)
- Head 3: base=160000 (near-content-based, minimal positional bias)

Implementation: new `rope_bases` field in GPTConfig, modified `_precompute_rotary_embeddings` to compute per-head cos/sin buffers with shape [1, T, n_head, d/2] instead of [1, T, 1, d/2]. The existing `apply_rotary_emb` handles this via broadcasting.

## Why
All heads currently share base=10000. While RoPE encodes multiple frequencies through different dimension channels, every head sees identical positional relationships for a given token pair. Multi-scale bases let heads specialize: head 0 for syntax/local structure (sharper positional signal), head 3 for semantic similarity (weaker positional bias). This is used in YaRN/LongRoPE literature for context extension and has been shown to improve attention quality.

Zero extra compute, zero extra parameters — only the precomputed rotary constants change.

## Risks
- Exp65 (base=1000 for ALL heads) regressed +0.014. Here only 1/4 heads is below 10000, and at a more conservative 2500.
- With 800 training steps, heads may not learn to specialize effectively.
- Worst case: ~0.002 regression. No crash/OOM risk.


## Résultats
- val_bpb: 1.077197
- peak_vram_mb: 10354.2
- status: IMPROVED
