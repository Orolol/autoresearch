Low-rank cross-block recurrent state for local attention layers with forgetting gate and zero-init output.

## What changed

In `train.py`, added a low-rank recurrent state mechanism to local attention layers (2-5) that propagates cross-block information:

1. **New parameters** (`CausalSelfAttention.__init__`, local layers only):
   - `state_proj`: Linear(128, 16) — shared Q/K projection into 16-dim state space
   - `state_decay`: per-head scalar — sigmoid gate controlling state forgetting
   - `state_gate`: per-head scalar — output gate (init=0 for neutral start)

2. **Modified forward** (`CausalSelfAttention.forward`, local_attn branch):
   After computing block-diagonal softmax attention (unchanged), adds cross-block context:
   - Project Q and K through shared low-rank projection (128 → 16 dim)
   - Loop over 8 blocks (fixed, compile-friendly):
     - Query accumulated state: `cross = q_low @ state` (B, H, BS, head_dim)
     - Update state: `state = decay * state + k_low^T @ v` (B, H, 16, 128)
   - Gate the cross-block contribution: `y = y + gate * cross`

3. **Init** (`GPT.init_weights`):
   - state_proj: uniform(-s, s) like other attention weights
   - state_decay: 0 → sigmoid(0)=0.5 (moderate forgetting)
   - state_gate: 0 → zero output (model starts equivalent to baseline)

Total new parameters: ~8.2K (4 local layers × 2056 params). Compute overhead: ~1.7%.

## Why

Three motivations:

1. **Fixes exp152's failures**: Cross-block linear recurrence (exp152, +0.006) used full-rank D=128 state → noisy, expensive, no forgetting. This uses d_state=16 (64x smaller), learnable decay, and zero-init gate.

2. **Clear architectural gap**: Local layers (2-5) have block_size=256 with hard boundaries. Information can only flow between blocks via the residual stream across layers (slow, 4 layers to traverse). The recurrent state provides direct cross-block flow within each layer.

3. **Structural change required**: exp171-173 were all HP tweaks. Partitioned Hyperconnections being tested in exp176 — this is an orthogonal direction.

## Key design choices

- **Low-rank (d_state=16)**: Acts as information bottleneck, capturing only the 16 most important KV modes. Keeps state matmuls at O(d_state × D) instead of O(D²).
- **Shared Q/K projection**: Single Linear(128, 16) applied to both Q and K. Reduces parameters by 2/3 vs separate projections.
- **Learnable decay**: `sigmoid(state_decay)` controls forgetting. Init 0.5 means moderate decay. Can learn to be near-0 (persistent memory) or near-1 (aggressive forgetting) per head.
- **Zero-init gate**: The cross-block contribution is multiplied by `state_gate` (init=0). At step 0, the model is exactly the baseline. The gate only grows if the cross-block signal improves the loss.

## Risks

- **Regression (medium)**: Exp152 suggests cross-block context may not help local layers. But exp152 had no decay and full-rank noise.
- **Crash (low)**: Fixed-length loop (8 iters), standard matmuls, no dynamic shapes. Should compile cleanly.
- **Neutral (medium-high)**: The residual stream already provides cross-layer information flow. Cross-block within a layer may be redundant.

Expected range: -0.003 to +0.006.


## Résultats (after crash fix)
- val_bpb: 1.066348
- peak_vram_mb: 14100.1
- status: discarded
