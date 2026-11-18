Dynamic attention temperature — input-dependent per-head Q scaling after QK-norm for adaptive attention sharpness.

## What changed

In `CausalSelfAttention.__init__` (train.py):
- Added `self.attn_temp = nn.Linear(self.gate_dim, self.n_head, bias=False)` — same shape as attn_gate (8→4), Muon-optimal near-square matrix.

In `CausalSelfAttention.forward` (train.py), after QK-norm:
- Compute `temp = 2 * torch.sigmoid(self.attn_temp(x[:, :, :self.gate_dim]))` — (B, T, n_head), range (0, 2), init at 1.0
- Scale Q: `q = q * temp.unsqueeze(-1)` — per-head per-position scaling before SDPA

In `GPT.init_weights` (train.py):
- Zero-init `attn_temp.weight` → sigmoid(0)=0.5 → temp=1.0 (neutral, identical to current behavior)

Total new parameters: 8 layers × (8 × 4) = 256 params (0.00036% of model).

## Why

QK-norm (q, k = norm(q), norm(k)) is essential for training stability but has a side effect: it removes ALL magnitude information from queries and keys. After QK-norm, every head at every position has the same attention "sharpness" — controlled solely by 1/sqrt(head_dim). This means:
- A position predicting a rare word continuation gets the same attention distribution shape as one predicting a common function word
- A head that needs to attend to one specific context token uses the same temperature as one that should average across many positions

Dynamic attention temperature restores per-token sharpness control in a learnable, input-dependent way.

**Key difference from dead qk_scales (exp67/68, +0.002)**: qk_scales was a STATIC per-head scalar — one number per head learned during training, same for all tokens. The dynamic temperature is a FUNCTION of the input (Linear(8→4) from first 8 dims), producing different temperatures for each token. This is the difference between adjusting the thermostat once vs. adjusting it separately for each room.

**Orthogonal to attention gate (exp141, -0.002)**: The gate modulates attention output MAGNITUDE (how much each head contributes to residual). Temperature modulates attention DISTRIBUTION SHAPE (sharp vs. broad). A head can have high gate + high temp (strong, focused contribution) or high gate + low temp (strong, averaged contribution).

Uses the same sparse input pattern (first gate_dim=8 dims) that proven effective for the attention gate.

## Risks

- **Neutral (medium)**: The model may not benefit from per-token sharpness control at 2048 context / 4 heads / ~830 steps.
- **Regression (very low)**: Zero-init ensures the model starts identical to current best. The temperature can only learn to help or stay neutral.
- **Crash (none)**: Standard nn.Linear + sigmoid + multiply. All shapes static. torch.compile compatible.
- **OOM (none)**: 256 new params + negligible activations.

Expected range: -0.003 to +0.003.

## If it works

- Tune temperature range: try `1 + sigmoid(...)` for range [1, 2] (only sharper, never broader)
- Try temperature on K as well (not just Q)
- Try wider temp input (gate_dim=12 or 16)
- Combine with removing qk_norm on Q only (let natural magnitude + learned temp replace norm)

## If it fails

- Per-token attention sharpness is not a bottleneck at this scale
- Remaining directions: FP8 matmuls (biggest "more steps" lever), DeltaNet/GLA (linear attention replacement)
- The architecture may be near its ceiling at ~830 steps on 2048 context


## Résultats
- val_bpb: 1.062436
- peak_vram_mb: 12503.3
- status: IMPROVED
