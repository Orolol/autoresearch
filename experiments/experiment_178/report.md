Sigmoid attention for local layers (2-5) — replace softmax with sigmoid gating for independent per-position attention control.

## What changed

In `train.py`, the `local_attn` branch of `CausalSelfAttention.forward` (lines 118-138):
- Replaced `F.scaled_dot_product_attention(q2, k2, v2, is_causal=True)` with manual sigmoid attention
- Compute scores via Q @ K^T (already QK-normed and temperature-scaled)
- Apply causal mask (triu, fill -1e4)
- Apply `torch.sigmoid(scores)` instead of softmax
- Normalize by causal position count (1, 2, ..., 256) to keep output magnitude stable

No new parameters. No new modules. Global layers (0,1,6,7) unchanged (still softmax SDPA).

## Why

Softmax attention forces a competition where weights must sum to 1 — each key "steals" attention from others. Sigmoid attention allows each position to independently gate its attention to each key, enabling patterns that softmax cannot represent (e.g., attending strongly to 5 positions simultaneously, or attending weakly to everything).

This is inspired by "Sigmoid Attention Is All You Need" (Apple, 2024) which showed sigmoid attention can match or beat softmax in certain regimes. Applied only to local layers (block_size=256) where:
1. The explicit attention matrix (256x256) is small enough for manual computation
2. Local context processing may benefit more from distributed attention patterns
3. Global layers keep proven softmax SDPA for efficiency

## Risks

- **Regression**: Sigmoid may converge slower — it must learn negative scores for selectivity, while softmax is selective by default
- **Throughput**: Loses flash attention kernel for local layers; manual matmul on 256x256 blocks should be <10% slower overall
- **Crash**: Very unlikely — all ops are standard and torch.compile friendly


## Résultats
- val_bpb: 1.082753
- peak_vram_mb: 13523.5
- status: discarded
