GLA-style input-dependent decay for chunkwise linear attention in local layers (2-5) — content-addressed forgetting replacing fixed exponential decay.

## What changed

In `train.py`, upgraded the chunkwise linear attention (exp179) from fixed per-head decay to GLA-style input-dependent decay:

1. **New parameter** (`CausalSelfAttention.__init__`, local layers only):
   - `decay_proj`: Linear(gate_dim=8, n_head=4, bias=False) — maps first 8 dims of input to per-head per-position decay modulation
   - Total: 4 layers x 32 params = 128 new parameters

2. **Modified forward** (`CausalSelfAttention.forward`, local_attn branch):
   - **Before (RetNet-style)**: `gamma = sigmoid(decay_rate)` — fixed per-head scalar, same decay at every position
   - **After (GLA-style)**: `gamma_t = sigmoid(decay_rate + decay_proj(x_t))` — per-position per-head decay computed from input content
   - Intra-chunk decay mask: `exp(cumsum(log_gamma)[i] - cumsum(log_gamma)[j])` via cumulative log-sum (numerically stable)
   - Cross-chunk decay: `exp(cumsum(log_gamma)[pos])` — each position applies its own accumulated decay to the recurrent state
   - State update: key decay and chunk decay computed from cumulative log-gamma

3. **Init** (`GPT.init_weights`):
   - `decay_rate = 4.0` (same as before) — base decay gamma ~ 0.982
   - `decay_proj.weight = 0` — zero modulation at init → exactly equivalent to exp179

## Why

1. **Priority #1 on queue**: "add GLA-style input-dependent decay" is explicitly listed as the next step.
2. **Key GLA insight**: Fixed decay treats all tokens equally — a punctuation mark decays the same as a subject noun. Content-dependent decay lets the model learn that important context tokens (subjects, verbs) should be retained longer while filler tokens (articles, punctuation) can be forgotten faster.
3. **Builds on the biggest win**: exp179 (-0.012) proved chunkwise linear attention works. This upgrades it with the mechanism that makes GLA outperform RetNet in the literature.
4. **Exact baseline equivalence at init**: zero-init decay_proj means gamma_t = sigmoid(4.0 + 0) = 0.982 everywhere — the model starts identical to exp179 and gradually learns content-dependent decay.

## Risks

- **Regression (low)**: Zero-init ensures neutral start. The model can only move away from exp179 baseline if it improves the loss.
- **Crash (low)**: Same loop structure as exp179. Decay masks become (B, H, BS, BS) instead of (H, BS, BS) but shapes are all static.
- **OOM (none)**: Extra ~17 MB per chunk for batch-dependent decay masks.
- **Compile time (medium)**: Slightly more complex graph due to cumsum/exp operations, but all standard ops.


## Résultats
- val_bpb: nan
- peak_vram_mb: 14578.5
- status: discarded
