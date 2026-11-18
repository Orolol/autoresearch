Dynamic key temperature — input-dependent per-head K scaling after QK-norm, complementary to Q-temp.

## What changed

In `CausalSelfAttention.__init__` (train.py):
- Added `self.key_temp = nn.Linear(self.gate_dim, self.n_head, bias=False)` — same shape as attn_temp (8→4), Muon-optimal near-square matrix.

In `CausalSelfAttention.forward` (train.py), after QK-norm:
- Renamed Q-temp variable to `q_temp` for clarity
- Added `k_temp = 2 * torch.sigmoid(self.key_temp(sparse_in))` — (B, T, n_head), range (0, 2), init 1.0
- Scale K: `k = k * k_temp.unsqueeze(-1)` — per-head per-position K scaling before SDPA

In `GPT.init_weights` (train.py):
- Zero-init `key_temp.weight` → sigmoid(0)=0.5 → temp=1.0 (neutral, identical to current behavior)

Total new parameters: 8 layers × (8 × 4) = 256 params (0.00036% of model).

## Why

Exp153 proved that input-dependent Q temperature works (-0.000644, new best 1.062436). Q-temp modulates how FOCUSED each query is — all keys are scaled uniformly for a given query position.

K-temp adds the complementary dimension: how PROMINENT each key position is. High K-temp makes a position more visible to all queries; low K-temp makes it less visible.

Together, Q-temp and K-temp create a rank-1 temperature matrix: the effective attention logit scale between query position i and key position j is `t_q[i] * t_k[j]`. This is fundamentally richer than Q-temp alone:
- Q-temp only: `softmax_j(t_q[i] * score[i,j])` — uniform scaling of all scores from query i
- Q+K temp: `softmax_j(t_q[i] * t_k[j] * score[i,j])` — per-key modulation within the softmax

Concrete examples of what K-temp enables:
- A rare/informative token can make its key more prominent (high t_k) to attract all queries
- A common function word can dampen its key (low t_k) to reduce attention noise
- Combined with Q-temp: a focused query (high t_q) attending to a prominent key (high t_k) gets extra-sharp attention

Uses the same sparse input pattern (first gate_dim=8 dims) proven effective for attention gate (exp141) and Q-temp (exp153).

## Risks

- **Neutral (medium)**: t_q * t_k may overparameterize the temperature space. The model might only need one axis of control.
- **Regression (very low)**: Zero-init ensures identical start. Both temps can stay at 1.0 if not useful.
- **Crash (none)**: Standard nn.Linear + sigmoid + multiply. All shapes static. torch.compile compatible.
- **OOM (none)**: 256 new params + negligible activations.

Expected range: -0.003 to +0.003.

## If it works

- Try asymmetric temperature ranges: Q-temp in [0.5, 1.5], K-temp in [0.8, 1.2] (more conservative K scaling)
- Try sharing the sparse input projection between Q-temp and K-temp (single Linear → 2*n_head, split)
- Try removing QK-norm on Q only (let natural magnitude + learned Q-temp replace norm)
- Try position-dependent temperature bias (add a learned per-position offset to temp)

## If it fails

- Per-key prominence control is redundant given attention gate and Q-temp
- The rank-1 temperature factorization doesn't add useful degrees of freedom
- Next directions: FP8 matmuls (biggest "more steps" lever), DeltaNet/GLA (full attention replacement)
- Consider that 1.062436 may be near the architecture's ceiling at ~830 steps


## Résultats
- val_bpb: 1.067878
- peak_vram_mb: 12509.1
- status: discarded
