Extend RWKV-style token shift to Q projection — full Q/K/V time mixing for all three attention projections.

## What changed

Extended token shift from K/V-only (exp126) to Q/K/V in `CausalSelfAttention`:

1. **`__init__`**: Added `self.time_mix_q = nn.Parameter(torch.empty(self.n_embd))` alongside existing `time_mix_k` and `time_mix_v`.

2. **`forward`**: Q projection now uses shifted input:
   - `mq = sigmoid(time_mix_q)` — per-channel mixing ratio for queries
   - `x_q = x * mq + x_prev * (1 - mq)` — mixed input for Q projection
   - `q = self.c_q(x_q)` instead of `q = self.c_q(x)`

3. **`init_weights`**: Added `time_mix_q.fill_(0.0)` so `sigmoid(0) = 0.5` = equal mix at initialization.

Total new params: 512 * 8 = 4,096 (0.005% of model). Zero compute overhead.

## Why

- **Direct follow-up on exp126** (-0.000326, new best): its report explicitly says "Apply token shift to Q as well" as the first next step
- **Full RWKV-style mixing**: In RWKV architectures, all projections (R/K/V/W/G) use separate time mixing. We had K/V only; this completes Q.
- **Self-correcting**: If Q mixing hurts, the model can learn `time_mix_q → +inf` (sigmoid → 1.0 = current token only), recovering exp126 behavior
- Queries become "what does this bigram want?" — local bigram context could help pattern matching
- Architectural change (priority 1), zero risk

## Risks

- **Neutral (likely)**: Q mixing may be redundant with full causal attention. Model might learn to not use it.
- **Regression (unlikely)**: Blurring query intent could slightly hurt attention precision.
- **Crash (none)**: Pure element-wise tensor ops.
- Expected range: -0.002 to +0.003

## If it works

- Try layer-dependent mixing initialization (early layers = more mixing, late layers = less)
- Try token shift for MLP input as well (separate time_mix_mlp parameter)
- Try mixing with position -2 in addition to -1 (multi-step lookback)

## If it fails

- Q shift is dead; K/V-only (exp126) is optimal. Queries need clean current-token representation.
- Try token shift for MLP input only (separate from attention shift)
- Focus on FP8 retry with simpler approach (torchao or manual fp8 for MLP only)


## Résultats
- val_bpb: 1.075740
- peak_vram_mb: 12109.2
- status: discarded
