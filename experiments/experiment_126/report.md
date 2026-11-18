RWKV-style token shift for attention K/V inputs — learnable per-channel mixing of current and previous token before K/V projections.

## What changed

Added RWKV-inspired "time mixing" to CausalSelfAttention. Before computing K and V projections, each token's representation is mixed with the previous token's representation using learnable per-channel sigmoid gates:

1. **CausalSelfAttention.__init__**: Added `time_mix_k` and `time_mix_v` as `nn.Parameter(torch.empty(n_embd))` — one per attention layer, per projection type.

2. **CausalSelfAttention.forward**: Before K/V projection:
   - `x_prev = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)` — shift right, repeat first token
   - `mk = sigmoid(time_mix_k)` — per-channel mixing ratio for keys
   - `mv = sigmoid(time_mix_v)` — per-channel mixing ratio for values
   - `x_k = x * mk + x_prev * (1 - mk)` — mixed input for K projection
   - `x_v = x * mv + x_prev * (1 - mv)` — mixed input for V projection
   - Q projection uses unshifted `x` (queries represent current token's intent)

3. **Initialization**: `time_mix_k/v = 0.0` so `sigmoid(0) = 0.5` = equal mix at start. Model learns optimal ratio per channel.

4. **Optimizer**: Time mix params (1D) separated from matrix params (2D, Muon). Added to AdamW with scalar_lr=0.5, betas=(0.96, 0.95).

Total new params: 2 * 512 * 8 = 8,192 (0.01% of model). Zero compute overhead (just element-wise ops).

## Why

- **RWKV-style recurrence** is explicitly listed as priority-1 architectural exploration in prompt.md
- Gives K/V projections access to local bigram context BEFORE attention computes it — the projections can produce richer key/value representations
- Q left unshifted: queries should represent "what the current token wants" without neighbor contamination
- First token uses self-shift (cat with itself) so no information loss at position 0
- Different from conv-before-MLP (exp107, +0.023, dead): MLP is position-independent so neighbor mixing hurts it, but attention is context-aggregation so enriching its inputs should help

## Risks

- **Regression (medium)**: Full causal attention already captures all pairwise relationships. The token shift provides redundant information that the model needs to learn to ignore.
- **Neutral (medium)**: The mixing might be too weak to matter given attention already sees everything.
- **Crash (none)**: Pure tensor operations, no new dependencies.
- Expected range: -0.003 to +0.005

## If it works

- Apply token shift to Q as well (full RWKV-style time mixing for Q/K/V)
- Try layer-dependent mixing initialization (deeper layers = less mixing, as they already have rich context)
- Try separate time_mix for attention vs MLP inputs (RWKV uses separate mixing for both sublayers)

## If it fails

- Token shift before attention is dead (like conv before MLP)
- Local context mixing is redundant with full causal attention at T=2048
- Focus remaining effort on FP8 retry (torchao or simpler approach) as the last "more steps" lever


## Résultats
- val_bpb: 1.067854
- peak_vram_mb: 11981.1
- status: IMPROVED
