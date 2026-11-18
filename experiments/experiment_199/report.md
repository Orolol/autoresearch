Pathway-specific attention gating for hybrid local layers — separate cross-chunk gate gives per-position control over recurrent vs local softmax contribution.

## What changed

In `train.py`, added a `cross_gate` module to `CausalSelfAttention` for local layers (2-5):

1. **New parameter** (`CausalSelfAttention.__init__`, local layers only):
   - `cross_gate`: `nn.Linear(gate_dim=8, n_head=4, bias=False)` — input-dependent per-head gating of the cross-chunk recurrent contribution. Uses the same sparse input pattern as the existing `attn_gate`.

2. **Modified forward** (`CausalSelfAttention.forward`, local_attn branch):
   - Old: `output = (intra + cross) * attn_gate` — combined gating, no way to differentiate pathways.
   - New: `output = (intra + cross_gate * cross) * attn_gate` — cross-chunk contribution is independently modulated before combining with intra-chunk softmax.
   - Intra and cross chunks tracked separately through the loop, then stacked and gated independently.

3. **Init** (`GPT.init_weights`):
   - `cross_gate` weights initialized to zero → sigmoid(0)=0.5 → ×2=1.0 → exactly equivalent to current behavior at step 0.

4. **Everything else unchanged**: global layers, SDPA path, state update, decay, all hyperparameters.

## Why

1. **Architectural limitation**: The current hybrid attention combines intra-chunk (softmax, sharp local) and cross-chunk (linear recurrence, smooth global) with a simple addition. The `attn_gate` gates the COMBINED output — it cannot suppress one pathway while keeping the other. The `attn_temp` scales Q, which affects BOTH pathways equally. Neither mechanism allows the model to say "for this token, trust local context but not distant context."

2. **Use case — document boundaries**: When the sequence crosses a document boundary, the recurrent state S contains stale context from the previous document. The model should suppress cross-chunk contribution here while maintaining strong intra-chunk attention. Currently impossible without also suppressing intra-chunk.

3. **Pivot from exhausted axes**: After 7 consecutive failures modifying the chunk attention mechanism itself (194-198), this changes HOW the two pathways are COMBINED rather than modifying either pathway.

4. **Minimal risk**: 128 new parameters total, neutral init, standard ops, compile-friendly. The worst case is neutral.

## Risks

- **Neutral (high)**: The existing attn_gate + attn_temp may already provide sufficient pathway control indirectly. 128 params in 830 steps may not differentiate.
- **Crash (very low)**: Standard nn.Linear, no new tensor shapes, no dynamic ops.
- **Regression (very low)**: Negligible compute overhead (one Linear(8,4) + sigmoid per local layer).


## Résultats
- val_bpb: 1.048702
- peak_vram_mb: 14384.8
- status: IMPROVED
