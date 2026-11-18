DeltaNet-style error-correcting state update for cross-chunk linear recurrence in local attention layers (2-5).

## What changed

In `train.py`, modified the cross-chunk state update rule in `CausalSelfAttention.forward` for local layers (2-5):

1. **New parameter** (`CausalSelfAttention.__init__`, local layers only):
   - `beta_rate`: Per-head learnable update rate, shape (n_head,). sigmoid(beta_rate) controls how aggressively the state corrects.

2. **Modified state update** (inside chunk loop):
   - Old (RetNet-style blind accumulation):
     ```
     S = decay * S + k^T @ v * scale
     ```
   - New (DeltaNet-style error correction):
     ```
     predicted_v = k @ S           # what state already predicts for these keys
     error = v - predicted_v       # novel information only
     S = decay * S + beta * k^T @ error * scale
     ```
   - The state only accumulates what it doesn't already know, preventing redundant KV associations from wasting finite state capacity.

3. **Everything else unchanged**: softmax intra-chunk attention, log-decay bias, cross-chunk query (Q @ S * cross_decay), all hyperparameters.

4. **Init**: `beta_rate = 4.0` → sigmoid ≈ 0.982. At step 0, S=0 → predicted_v=0 → error=v → nearly identical to current baseline.

## Why

1. **Fundamentally different from all prior chunk attention experiments**: exp183 (conv), exp189 (hyperconnections), exp190 (multi-scale decay), exp191 (GLA input-dependent decay), exp192 (extend layers), exp193 (hybrid softmax), exp194 (chunk_size), exp195 (memory tokens) — ALL modified intra-chunk, layer assignment, or added mechanisms. NONE touched the core state update rule.
2. **DeltaNet validated in literature**: Yang et al. 2024 ("Gated Delta Networks") showed error-correcting updates significantly outperform RetNet's blind accumulation by using state capacity more efficiently.
3. **State capacity bottleneck**: S is D×D = 128×128 = 16K entries per head. With blind accumulation over 2048 tokens, redundant associations waste capacity. Error correction naturally deduplicates.
4. **Minimal overhead**: 16 new params total (4 heads × 4 layers). One extra matmul (B,H,BS,D)×(B,H,D,D) per chunk = ~same cost as existing Q@S. Estimated 5-10% total compute overhead.
5. **Near-neutral init**: No regression risk at step 0.

## Risks

- **Regression (medium)**: Extra matmul per chunk costs ~40-80 training steps. State quality must compensate.
- **Crash (very low)**: Standard ops, static shapes, compile-friendly.
- **Instability (low)**: Error correction could amplify noise, but exponential decay bounds the state.


## Résultats
- val_bpb: 1.064450
- peak_vram_mb: 14831.3
- status: discarded
