RoPE-free cross-chunk recurrence + multi-scale decay init for hybrid local attention layers (2-5).

## What changed

In `train.py`, two complementary changes to cross-chunk linear recurrence in local layers (2-5):

1. **RoPE-free cross-chunk path** (`CausalSelfAttention.forward`, local layers):
   - Compute pre-RoPE normalized Q/K (`q_raw`, `k_raw`) before applying rotary embeddings
   - Intra-chunk softmax attention uses RoPE-encoded Q/K (unchanged — positional information is useful for local token selection)
   - Cross-chunk recurrence uses RoPE-free Q/K: `cross = q_raw @ S * cross_decay` and `S = decay * S + k_raw^T @ V * scale`
   - Dynamic attention temperature applied to both RoPE and raw Q paths

2. **Multi-scale decay initialization** (`GPT.init_weights`):
   - Old: uniform `decay_rate = 4.0` for all heads → sigmoid ≈ 0.982, ~55-token effective context
   - New: per-head `decay_rate = [2.0, 3.5, 5.0, 7.0]` → sigmoid ≈ [0.88, 0.97, 0.993, 0.999]
   - Effective contexts: ~8, ~33, ~143, ~1000 tokens per head
   - Heads specialize at different temporal scales (like multi-scale RoPE does for global layers)

3. **No new parameters**: RoPE-free path reuses existing Q/K projections (just skips rotary embedding). Multi-scale init changes initial values only (still learnable).

## Why

1. **RoPE interference in accumulated state**: The recurrent state S accumulates K^T @ V products from past chunks. When K has RoPE, keys at different absolute positions create position-dependent angle patterns that get mixed in S. A future query Q with its own RoPE angles creates cosine interference patterns when querying S — the retrieval quality depends on absolute position differences rather than semantic similarity. Without RoPE on the cross-chunk path, retrieval is purely content-based + decay-weighted (which is what we want).

2. **Different from exp190**: Exp190 removed RoPE from BOTH intra-chunk AND cross-chunk in local layers (+0.003 regression). Here we KEEP RoPE for intra-chunk softmax (where relative position matters for token selection) and ONLY remove it from cross-chunk (where it causes interference). This is a strictly different and more targeted change.

3. **Multi-scale decay isolated from exp190**: Exp190 combined multi-scale decay + RoPE-free and got +0.003. We can't tell which caused the regression. This experiment isolates multi-scale decay (which was the priority queue recommendation) from full RoPE removal.

4. **Complementary changes**: Multi-scale decay gives heads different temporal receptive fields. RoPE-free cross-chunk gives cleaner semantic retrieval. Together they make each head's cross-chunk pathway both specialized (temporal scale) and precise (no positional noise).

## Risks

- **Regression (medium)**: Cross-chunk Q without RoPE loses positional information about the query's location in the sequence. The decay already handles temporal recency, but some position-dependent retrieval patterns may be lost.
- **Crash (very low)**: All ops are standard (norm, transpose, matmul). Shapes are static and identical to baseline. Compile-friendly.
- **OOM (very low)**: Extra q_raw/k_raw tensors add ~2MB per local layer. Well within budget.
- **Interaction risk (low-medium)**: Two changes at once make diagnosis harder if the result is bad. However, both are small, targeted, and affect only the cross-chunk path.


## Résultats
- val_bpb: 1.060267
- peak_vram_mb: 14870.8
- status: discarded
