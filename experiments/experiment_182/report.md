Multi-scale decay initialization + chunk_size=512 for chunkwise linear attention in local layers (2-5).

## What changed

In `train.py`, two changes to the chunkwise linear attention mechanism (exp179 baseline):

1. **chunk_size 256 → 512** (`CausalSelfAttention.__init__`, line 88):
   - Intra-chunk attention now covers 512 tokens (up from 256) — better local attention quality
   - 4 chunks per sequence instead of 8 — fewer cross-chunk recurrent state transitions, less error accumulation
   - Trade-off: ~67% more compute for local attention (4 * 512^2 vs 8 * 256^2), but still much cheaper than global SDPA (2048^2)

2. **Multi-scale decay init** (`GPT.init_weights`, line 287):
   - Before: `decay_rate.fill_(4.0)` → all heads gamma ≈ 0.982 → ~55-token effective context
   - After: `decay_rate = [2.0, 3.5, 5.0, 7.0]` → gamma ≈ [0.88, 0.97, 0.99, 0.999]
   - Effective context per head: [8, 31, 143, 1000] tokens
   - Head 3 (gamma ≈ 0.999) has ~1000-token effective context through recurrent state, covering most of the 2048-token sequence
   - Analogous to multi-scale RoPE (exp80) which gave per-head specialization across frequency scales

## Why

1. **Priority queue directive**: After exp181 NaN, the queue recommends chunk_size=512 and multi-scale decay init
2. **Multi-scale RoPE analogy**: Per-head base frequencies worked (exp80, kept). Per-head decay rates is the temporal analog for linear attention
3. **Better intra-chunk quality**: 512 tokens of exact causal attention captures more local structure than 256
4. **Fewer recurrent transitions**: 4 state updates vs 8 → less error accumulation in the cross-chunk recurrent state
5. **Head specialization without complexity**: Different temporal scales at init, learnable — model can adjust

## Risks

- **Regression (medium)**: More intra-chunk compute → potentially fewer training steps → could offset quality gains
- **Crash (very low)**: Same code path as exp179, just different constants
- **Neutral (medium)**: If uniform decay was already near-optimal, multi-scale init may not matter
