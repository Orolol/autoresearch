Increase chunk_size from 256 to 512 for hybrid local attention layers (2-5) — larger softmax intra-chunk windows + fewer cross-chunk recurrence steps.

## What changed

In `train.py`, line 88, changed `self.block_size = 256` to `self.block_size = 512`.

This doubles the chunk size for chunkwise hybrid attention in local layers (2-5). Each chunk now processes 512 tokens with full causal softmax attention + learned log-decay positional bias (exp193's hybrid approach), and the sequence is split into 4 chunks instead of 8.

## Why

1. **Explicitly untested**: Priority queue lists chunk_size=512 as the untested next step (exp182 INFRA crash, code never ran on GPU). Crash analysis confirmed it was an infrastructure issue, not a code bug.
2. **Synergy with exp193's hybrid softmax**: Exp193's key insight was that softmax intra-chunk >> linear intra-chunk. Larger chunks amplify this advantage — 512-token softmax windows give much sharper, more precise attention than 256-token windows. The model can attend to exactly the right token among 512 candidates instead of 256.
3. **Fewer recurrence steps**: 4 cross-chunk transitions instead of 8. Each transition through the recurrent state is lossy (exponential decay). Halving transitions means less information loss for long-range context, especially for slow-decay heads.
4. **Better intra-chunk coverage**: 512 tokens = 25% of the full 2048 context within each chunk. With cross-chunk recurrence, the model effectively has high-quality full-context coverage.
5. **Clean single-variable change**: Only one number changed, easy to diagnose effect vs exp193 baseline.

## Risks

- **Regression (medium)**: Intra-chunk softmax attention compute is O(BS^2) = O(512^2) vs O(256^2), a 4x per-chunk increase. With 4 chunks instead of 8, total intra-chunk FLOPs are ~2x higher. This only affects 4 local layers (global layers unchanged), so total model compute increase is maybe 5-10%. May lose ~40-80 training steps.
- **Crash (very low)**: Same code path, T=2048 is divisible by 512. All shapes remain static.
- **OOM (low)**: Softmax attention matrix per chunk is (B, H, 512, 512) instead of (B, H, 256, 256) = 4x larger per chunk. But only one chunk processed at a time. Peak memory increase ~64MB, well within budget.
- **Neutral (medium)**: If 256-token chunks were already capturing all relevant local patterns, the extra precision doesn't help and the step count reduction dominates.


## Résultats
- val_bpb: 1.052590
- peak_vram_mb: 15058.2
- status: discarded
