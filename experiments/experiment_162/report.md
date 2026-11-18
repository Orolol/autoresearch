Value centroid injection — inject causal running mean of V into attention output via learned per-head gate for global context in local attention layers.

## What changed

In `CausalSelfAttention.__init__` (train.py):
- Added `self.centroid_gate = nn.Linear(self.gate_dim, self.n_head, bias=False)` — same shape as attn_gate and attn_temp (8->4), Muon-compatible.

In `CausalSelfAttention.forward` (train.py):
- After VE addition and before the attention branch (full/local/FA3): compute `v_centroid = cumsum(v, dim=1) / arange(1, T+1)` — the causal running mean of V across positions.
- After the attention gate and before reshape/c_proj: add `c_gate.unsqueeze(-1) * v_centroid` to y, where `c_gate = self.centroid_gate(x[:, :, :gate_dim])`.
- The gate has no activation function (raw linear output), so it can be positive or negative. Zero-init means no centroid at start (neutral).

In `GPT.init_weights` (train.py):
- Zero-init `centroid_gate.weight` (gate starts at 0, no centroid injection at init).

Total new parameters: 8 layers x (8 x 4) = 256 params (0.00036% of model).
Compute change: negligible (cumsum + divide + multiply + add per layer).
Memory change: +1 intermediate tensor per layer (v_cumsum), ~64MB total (~0.5% of 12GB peak).

## Why

Local attention layers (2-5, block_size=256) are blind beyond their block. Cross-block information currently comes only from:
1. Bookend full-attention layers (0, 1, 6, 7) via residual stream
2. Token shift (1-step K/V lookback)
3. U-Net skip connections

The value centroid provides a NEW information pathway: the uniform causal average of all V vectors up to position t. This is:
- **Global**: summarizes the entire sequence, not just a 256-token block
- **Cheap**: O(T) cumsum, no attention computation
- **Content-aware**: V reflects actual token content (unlike positional encodings)
- **Per-head**: each head gets its own centroid from its own V stream
- **Gated**: model learns when to use it (local layers benefit, full-attention layers can suppress it)

This is different from all dead ideas:
- Token shift (exp126): 1-step K/V mixing. Centroid is full-sequence V summary.
- Cross-block linear recurrence (exp152, +0.006): propagated attention KV state across blocks. Centroid uses V directly (before attention), no recurrence.
- Hybrid local/global heads (exp159, +0.002): gave 1 head full attention in local layers. Centroid provides global info to ALL heads at near-zero cost.

The key insight: the centroid is orthogonal to attention. Attention computes a RELEVANCE-WEIGHTED average. The centroid is a UNIFORM average. They capture different statistics of the value stream.

## Risks

- **Neutral (high)**: The bookend layers' full attention may already provide sufficient global context via the residual stream, making the centroid redundant.
- **Regression (low)**: The uniform average lacks relevance filtering. For positions where specific context matters (most of them), adding an unweighted average could inject noise. However, the gate starts at 0 (neutral) and the model can learn to suppress it.
- **Memory (low)**: v_cumsum/v_centroid add ~64MB. Peak VRAM is 12.2GB with 32GB available.
- **Crash (none)**: cumsum, arange, division, multiply are all standard ops. All shapes static. torch.compile compatible.
- **OOM (none)**: negligible memory increase.

Expected range: -0.003 to +0.002.

## If it works

- Try gated centroid with sigmoid (constrain to non-negative mixing)
- Try centroid on K as well (centroid key for content-similarity matching)
- Try exponential moving average instead of uniform mean (recent tokens weighted more)
- Try centroid only on local attention layers (skip for full-attention bookend layers)

## If it fails

- Uniform averages are too noisy — the model can't extract useful signal from unweighted V means
- Full-attention bookend layers already provide sufficient global context
- Next directions: FP8 matmuls (biggest "more steps" lever), DeltaNet/GLA (full attention replacement)
- Consider that 1.062436 may be near the architecture's practical ceiling at ~830 steps


## Résultats
- val_bpb: 1.070433
- peak_vram_mb: 13533.1
- status: discarded
