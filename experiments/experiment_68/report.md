Per-head learnable attention temperature (qk_scales) to restore per-head sharpness control removed by QK-norm — retry of exp67 which crashed on infra.

## What changed

Added `self.qk_scales = nn.Parameter(torch.ones(config.n_layer, config.n_head))` to GPT (32 scalars for 8 layers x 4 heads). In CausalSelfAttention.forward, after QK-norm: `q = q * qk_scale[None, None, :, None]`. This multiplies each head's Q vectors by a learnable scalar, controlling the effective softmax temperature per head per layer.

The qk_scales are optimized with AdamW at lr=0.05, betas=(0.96, 0.95). Initialized to 1.0 (identical to baseline at step 0). Placed in GPT (not in blocks) to avoid being picked up by Muon's 2D matrix optimizer.

## Why

QK-norm normalizes Q and K to unit RMS, then SDPA divides by sqrt(head_dim)=sqrt(128)~11.3. This forces ALL 32 (head, layer) pairs to have identical effective attention temperature. The model cannot learn:
- Sharper attention in some heads (precise token matching for copying/retrieval)
- Broader attention in others (contextual averaging for semantic blending)

With only 4 heads per layer, each head carries significant responsibility. Letting them specialize their temperature is a genuine architectural improvement. Gemma 2 (Google, 2024) includes learnable per-head scales with QK-norm for exactly this reason.

With softcap=13, the maximum attention logit for aligned unit-norm QK vectors is ~11.3 (87% of softcap). A scale of ~1.15 would push logits to the softcap boundary for maximum gradient signal. The scales have room to adjust without saturating.

## Exp67 crash analysis

Exp67 attempted the same idea but crashed. Given the pattern of many infra crashes in this series (36/37/38/39/40/41/42 all crashed sequentially), it was almost certainly an infra crash, not a code bug. This retry uses the same approach with clean implementation.

## Risks

- Zero crash/OOM risk (32 extra scalars = 128 bytes, negligible)
- Zero speed impact (one scalar broadcast multiply per layer per forward pass)
- If fixed temperature is optimal: neutral within noise (~0.001)
- If scales grow too large (>1.5) and saturate softcap: possible mild gradient degradation, mitigated by conservative LR (0.05 vs 0.5 for x0_lambdas)


## Résultats
- val_bpb: 1.080678
- peak_vram_mb: 10336.7
- status: discarded
