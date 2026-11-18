Extend chunkwise linear attention from layers 2-5 to layers 1-6 — VE-bookend-only global attention for compute savings and more training steps.

## What changed

In `train.py`, line 88, changed the local attention condition from:
- Old: `self.local_attn = (layer_idx >= 2 and layer_idx < n - 2)` (layers 2,3,4,5 = 4 linear, 4 global)
- New: `self.local_attn = (layer_idx >= 1 and layer_idx < n - 1)` (layers 1,2,3,4,5,6 = 6 linear, 2 global)

Only the VE bookend layers (0 and 7) retain full softmax attention. Layers 1 and 6 switch from O(T^2) softmax SDPA to chunkwise linear attention with learned per-head exponential decay and cross-chunk recurrent state.

## Why

1. **All mechanism improvements failed**: exp183 (conv +0.006), exp189 (hyperconnections +0.006), exp190 (multi-scale decay + RoPE-free +0.003), exp191 (GLA input-dependent decay +0.001). Instead of enhancing the mechanism, extend its reach.
2. **Compute savings**: Removing 2 full softmax attention layers saves ~12% of total attention FLOPs. At ~830 steps/5min, this could yield 80-100 extra steps.
3. **Cross-chunk recurrence**: Unlike block-diagonal SDPA (exp150, neutral), chunkwise linear attention propagates a recurrent KV state across chunks, providing effective full-sequence context for layers 1 and 6.
4. **Different from exp180**: ALL layers linear was catastrophic (+0.044) because VE layers 0,7 lost softmax. Here VE layers keep full softmax.
5. **Minimal change**: Single-line modification, clean AB-test signal.

## Risks

- **Regression (medium)**: Layers 1 and 6 are adjacent to VE bookend layers and may need softmax's precise attention for processing rich VE-augmented representations.
- **Crash (very low)**: Same code paths already running for layers 2-5, just 2 more layers using them.
- **OOM (very low)**: Linear attention uses less memory than softmax SDPA.
- **Neutral (medium)**: Compute savings may be offset by slightly lower quality from layers 1,6 losing softmax precision.


## Résultats
- val_bpb: 1.054170
- peak_vram_mb: 14927.3
- status: discarded
