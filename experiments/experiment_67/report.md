Per-head learnable attention temperature (qk_scales) to let each head and layer optimize its own attention sharpness, overcoming a limitation of QK-norm.

## What changed

Added `self.qk_scales = nn.Parameter(torch.ones(n_layer, n_head))` to GPT. In CausalSelfAttention.forward, after QK-norm: `q = q * qk_scale[None, None, :, None]`. This multiplies each head's Q vectors by a learnable scalar, effectively controlling the softmax temperature per head per layer.

The qk_scales are optimized with AdamW at lr=0.05, betas=(0.96, 0.95). Initialized to 1.0 (identical to current behavior at step 0). 32 total scalars (8 layers x 4 heads).

## Why

QK-norm normalizes Q and K to unit RMS, then SDPA divides by sqrt(head_dim)=sqrt(128). This forces ALL heads at ALL layers to have the same effective attention temperature (~1/11.3). The model cannot learn:
- Sharper attention in some heads (precise token matching, e.g. for copying/retrieval)
- Broader attention in others (contextual averaging, e.g. for semantic blending)

With softcap=13, the maximum attention logit for aligned vectors is ~11.3 (87% of softcap), leaving headroom for sharper attention without saturating the tanh. A scale of ~1.15 would push logits to the softcap boundary for maximum gradient signal.

This is a genuine architectural fix, not a hyperparameter tweak — it restores a degree of freedom that QK-norm removes. Modern architectures with QK-norm (Gemma 2) include learnable per-head scales for exactly this reason.

## Risks

- Zero crash/OOM risk (32 extra scalars, negligible memory)
- Zero speed impact (one scalar broadcast multiply per layer)
- If fixed temperature is optimal: neutral within noise
- If scales grow too large (>1.5) and saturate softcap: possible mild gradient degradation. The conservative LR (0.05 vs 0.5 for x0_lambdas) mitigates this.
