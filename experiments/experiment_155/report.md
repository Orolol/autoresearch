Per-head input-dependent V scaling — supply-side value magnitude modulation complementary to demand-side attention gate.

## What changed

In `CausalSelfAttention.__init__` (train.py):
- Added `self.val_scale = nn.Linear(self.gate_dim, self.n_head, bias=False)` — same shape as attn_gate and attn_temp (8->4), Muon-optimal near-square matrix.

In `CausalSelfAttention.forward` (train.py), after VE injection and before RoPE/SDPA:
- Compute `v_s = 2 * torch.sigmoid(self.val_scale(x[:, :, :self.gate_dim]))` — (B, T, n_head), range (0, 2), init at 1.0
- Scale V: `v = v * v_s.unsqueeze(-1)` — per-head per-position V magnitude modulation

In `GPT.init_weights` (train.py):
- Zero-init `val_scale.weight` -> sigmoid(0)=0.5 -> scale=1.0 (neutral, identical to current behavior)

Total new parameters: 8 layers x (8 x 4) = 256 params (0.00036% of model).

## Why

The model has two existing per-head modulation mechanisms, both operating on the QUERY/OUTPUT side:
- **Attention gate** (exp141, -0.002): `output *= gate[query_pos]` — controls how much each query RECEIVES
- **Q-temperature** (exp153, -0.0006): scales Q magnitude to sharpen/broaden attention distribution

Neither controls the KEY/VALUE side. V-scaling adds supply-side modulation:
- **Val_scale**: `v[key_pos] *= scale[key_pos]` — controls how much each key PROVIDES

Mathematically: `final[i] = gate[i] * sum_j softmax(q[i]·k[j]) * scale[j] * v[j]`

Gate[i] scales the entire weighted sum uniformly (same scale for all attended values). Scale[j] scales each term inside the sum independently (different scale per attended position). These provide genuinely different capabilities:
- A position can attract high attention (through Q·K) but contribute little content (low V-scale)
- A position can have low attention weight but amplified content (high V-scale)

**Key difference from dead K-temperature (exp154, +0.005)**: K-temp changed the attention DISTRIBUTION (which keys get attended to) — redundant with Q-temp which already controls distribution shape. V-scale changes value CONTENT magnitude without affecting the distribution. Orthogonal mechanism.

Uses the same sparse input pattern (first gate_dim=8 dims) proven in attention gate (exp141) and Q-temp (exp153).

## Risks

- **Neutral (medium)**: c_v projection already provides linear value scaling as a function of input. The multiplicative sigmoid modulation adds nonlinear control but may not be needed.
- **Redundancy (low-medium)**: Combined with attn_gate, the model now has both input-side (val_scale) and output-side (gate) per-head modulation. If the interaction is fully captured by one of them, the other is wasted.
- **Regression (very low)**: Zero-init guarantees identical start to current best.
- **Crash (none)**: Standard nn.Linear + sigmoid + multiply. All shapes static. torch.compile compatible.
- **OOM (none)**: 256 new params + negligible activations.

Expected range: -0.003 to +0.003.

## If it works

- Try asymmetric val_scale range: `1 + sigmoid(x)` (range 1-2, only amplify, never suppress)
- Try val_scale from token-shifted input (x_v features) instead of current position features
- Try combining val_scale and attn_gate into a single wider projection (shared input -> split output)
- Try val_scale only on VE layers (0, 7) where value content is enriched

## If it fails

- Supply-side per-head modulation is not useful — the model prefers query-side control
- The c_v projection already provides sufficient value magnitude control
- Next directions: FP8 matmuls (biggest "more steps" lever), DeltaNet/GLA (full attention replacement)
- Consider that 1.062436 may be near the architecture's ceiling at ~830 steps


## Résultats
- val_bpb: 1.068479
- peak_vram_mb: 13020.1
- status: discarded
