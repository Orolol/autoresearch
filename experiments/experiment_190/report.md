Multi-scale decay init + RoPE-free linear attention for local layers (2-5) — head specialization via diverse temporal horizons and removal of redundant positional encoding.

## What changed

Two changes to the chunkwise linear attention in local layers (2-5):

1. **Multi-scale decay init** (`GPT.init_weights`):
   - Replaced uniform `decay_rate.fill_(4.0)` with per-head `[2.0, 3.5, 5.0, 7.0]`
   - sigmoid([2.0, 3.5, 5.0, 7.0]) = [0.88, 0.97, 0.993, 0.999]
   - Effective context: ~[8, 29, 143, 1000] tokens per head
   - Head 0: very local n-gram patterns; Head 3: near-full sequence via recurrence
   - Analogous to multi-scale RoPE (exp80, kept) but for the decay mechanism

2. **Remove RoPE from local layers** (`CausalSelfAttention.forward`):
   - Added `if not self.local_attn:` guard around `apply_rotary_emb` calls
   - Local layers (2-5) no longer apply rotary position embeddings to Q/K
   - Global layers (0,1,6,7) keep RoPE + softmax unchanged
   - QK-norm still applied to all layers

## Why

1. **Multi-scale decay** is priority queue #1 next step — exp182 tested this but INFRA-crashed (never GPU-tested). Crash analysis confirms code was fine.
2. **RoPE removal** follows RetNet/RWKV design: linear attention with decay is a self-contained positional encoding system. RoPE adds a second, potentially conflicting positional signal. Without RoPE, Q@K^T becomes purely content-based similarity, and all position info comes from the decay gamma^(i-j). This is how RetNet, RWKV, and GLA work.
3. **Synergy**: Multi-scale decay replaces multi-scale RoPE's function (head specialization across temporal horizons) for local layers. Each head gets a different "memory length" instead of a different "frequency resolution."
4. **Compute savings**: Skipping rotary embedding computation for 4 layers reduces per-step FLOPs slightly.
5. **All failed alternatives exhausted**: conv on Q/K (exp183, +0.006), partitioned hyperconnections (exp189, +0.006), GLA input-dependent decay (exp181, NaN).

## Risks

- **Regression (medium)**: RoPE removal is the risky part. If intra-chunk attention needs position discrimination beyond what decay provides (e.g., two identical tokens at different positions), quality may suffer.
- **Crash (very low)**: Only a conditional skip + different init constants. No new ops, shapes unchanged.
- **Neutral (medium)**: If decay already provides sufficient position and multi-scale doesn't help, both changes are neutral.


## Résultats
- val_bpb: 1.052605
- peak_vram_mb: 14120.8
- status: discarded
