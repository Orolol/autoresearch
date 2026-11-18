RetNet-style hybrid attention for local layers — softmax intra-chunk + linear recurrence cross-chunk with learned log-decay positional bias.

## What changed

In `train.py`, replaced the all-linear intra-chunk attention in local layers (2-5) with softmax causal attention, keeping cross-chunk linear recurrence unchanged:

1. **Removed**: `intra_decay` precomputation (gamma^(i-j) * causal mask for linear scores)
2. **Added**: `causal_bool = (rel >= 0)` — boolean mask for softmax causal masking
3. **Added**: `log_decay_bias = rel * log(gamma)` — per-head learned positional bias in log-space (ALiBi-like)
4. **Changed intra-chunk computation**:
   - Old: `scores = QK^T * scale * intra_decay` → linear attention with multiplicative decay
   - New: `scores = QK^T * scale + log_decay_bias; masked_fill(-inf); softmax(scores)` → softmax attention with additive decay bias
5. **Unchanged**: cross-chunk recurrence (Q @ S * cross_decay), state update (decay * S + K^T V * scale), all precomputed cross/key/chunk decay masks

## Why

1. **Softmax vs linear for local context**: Within a 256-token chunk, softmax produces sharp, selective attention distributions (can attend to exactly 1 token). Linear attention distributes weight more uniformly (QK similarity * decay), which blurs local patterns.
2. **RetNet design validation**: RetNet (Sun et al. 2023) and HGRN2 use exactly this hybrid: softmax for intra-chunk quality, linear recurrence for inter-chunk efficiency. Our current all-linear approach is the "pure recurrent" variant, which RetNet found inferior to the hybrid.
3. **All linear-attention improvements exhausted**: exp183 (conv +0.006), exp189 (hyperconnections +0.006), exp190 (multi-scale decay + RoPE-free +0.003), exp191 (GLA input-dependent decay +0.001), exp192 (extend to layers 1-6 +0.004). Six consecutive failures to improve the mechanism → change the paradigm.
4. **Log-decay bias**: Converts the multiplicative per-head decay (gamma^distance) into an additive log-space bias for softmax logits. This preserves the learned distance weighting from exp179, now acting as a learned ALiBi-like slope per head. At init (gamma≈0.982), bias ≈ -0.018 per position distance — mild locality preference.
5. **Minimal change**: 4 lines changed, same parameters, same shapes, same cross-chunk logic.

## Risks

- **Regression (medium)**: Magnitude mismatch between softmax intra (output ≈ V, weights sum to 1) and linear cross (output ≈ D^-0.5 * V). The cross-chunk contribution may be relatively smaller. However, this might actually HELP by making the model rely more on precise local attention from softmax.
- **Crash (very low)**: All ops (softmax, masked_fill, log) are standard, shapes are static, torch.compile handles these cleanly.
- **Compile overhead (low)**: softmax + masked_fill per chunk may slightly increase compile graph complexity. First few steps may be slower.
- **Neutral (medium)**: If all-linear was already capturing the right patterns within chunks, softmax doesn't add value and the overhead costs steps.


## Résultats
- val_bpb: 1.049078
- peak_vram_mb: 14126.8
- status: IMPROVED
