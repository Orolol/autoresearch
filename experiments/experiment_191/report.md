GLA-style input-dependent decay for chunkwise linear attention in local layers (2-5) — content-addressed forgetting with numerically stable implementation.

## What changed

In `train.py`, replaced the fixed per-head exponential decay in chunkwise linear attention with GLA-style input-dependent per-position decay:

1. **New parameter** (`CausalSelfAttention.__init__`, local layers only):
   - `decay_proj`: Linear(gate_dim=8, n_head=4, bias=False) — maps sparse input dims to per-head decay modulation
   - Total: 4 local layers x 32 params = 128 new parameters (negligible)

2. **Modified forward** (`CausalSelfAttention.forward`, local_attn branch):
   - Old: `gamma = sigmoid(decay_rate)` — fixed per-head decay, same for all positions
   - New: `gamma_t = sigmoid(decay_rate + decay_proj(x[:, :, :gate_dim]))` — per-position, per-head decay
   - Each position decides its own forget rate based on its content
   - Decay masks computed per-chunk inside the loop (no longer pre-computable)
   - Numerically stable: gamma clamped to [0.8, 1.0], log-cumsum in float32
   - This is the key fix for exp181's NaN: clamping prevents log(0) = -inf

3. **Init** (`GPT.init_weights`):
   - `decay_proj.weight` zero-initialized → at step 0, gamma = sigmoid(4.0) ≈ 0.982 for all positions
   - Exactly equivalent to exp179/185 baseline at init

## Why

1. **Priority queue #1**: GLA input-dependent decay is the top recommended direction. Exp181 tried it but NaN'd due to unclamped `cumsum(log_gamma)`.
2. **Content-addressed memory**: Fixed decay forgets at a constant rate regardless of content. GLA lets the model retain important tokens (low decay) and forget irrelevant ones (high decay). This is the fundamental innovation from GLA (Yang et al. 2024) over RetNet.
3. **Builds on biggest win**: exp179's chunkwise linear attention was -0.012, the largest single-experiment gain in 190 experiments. GLA is the natural next step.
4. **Numerically stable**: Clamping gamma >= 0.8 before log ensures log values in [-0.223, 0], cumsum stays bounded, and exp never overflows/underflows in float32. With BS=256, minimum decay is 0.8^256 ≈ 2.7e-25, well above float32 minimum.
5. **Zero regression risk at init**: decay_proj is zero → identical to current fixed-decay baseline.

## Risks

- **Regression (low-medium)**: Per-chunk decay computation adds ~5-10% overhead per local layer (cumsum + exp of BS x BS matrix per chunk). May lose 1-2 training steps.
- **Crash (low)**: All ops are standard (log, cumsum, exp, matmul). Shapes are static. torch.compile should handle it.
- **Memory (low)**: intra_decay is now (B, H, BS, BS) per chunk instead of (H, BS, BS), but only one chunk at a time. ~16 MB overhead.
- **Compile time (medium)**: More complex per-chunk computation graph. First few steps may be slower.
- **NaN (very low)**: gamma clamped to [0.8, 1.0], all intermediate values bounded. Should not reproduce exp181's instability.


## Résultats
- val_bpb: 1.050692
- peak_vram_mb: 14621.8
- status: discarded
