Extend chunkwise linear attention to ALL 8 layers, eliminating softmax entirely from the model.

## What changed

In `train.py`, two minimal changes:

1. **`CausalSelfAttention.__init__`** (line 87): Changed `self.local_attn = (layer_idx >= 2 and layer_idx < n - 2)` to `self.local_attn = True`. All 8 layers now use chunkwise linear attention with decay instead of softmax SDPA.

2. **`GPT.init_weights`** (line 287): Removed the `if block.attn.local_attn:` guard on `decay_rate` initialization. All layers get `decay_rate = 4.0` (gamma ~ 0.982).

Total new parameters: 16 scalars (4 heads x 4 newly-converted layers). VE layers (0, 7) keep their value embeddings unchanged.

## Why

1. **Builds on the biggest breakthrough**: exp179 showed -0.012 BPB by replacing softmax with chunkwise linear attention in layers 2-5. This extends the winning mechanism to all layers.

2. **Massive compute savings**: The 4 global layers currently do O(2048^2) softmax attention per head. Converting to 256-token chunkwise linear attention reduces to O(8 * 256^2) = ~7.6x less attention FLOPs per global layer. This should yield significantly more training steps in 5 minutes.

3. **Full-sequence context preserved**: The recurrent KV state with learned exponential decay provides full 2048-token context for all layers, so the model doesn't lose long-range capability.

4. **Radical simplification**: The model becomes a uniform architecture — all layers use the same attention mechanism. No more softmax/linear split.

## Risks

- **Regression (medium)**: Global layers may need softmax's exact normalized attention for long-range reasoning. The exponential decay in the recurrent state could lose important long-distance information.
- **Crash (very low)**: Same code path as exp179, just activated for more layers.
- **OOM (none)**: Linear attention uses less memory than SDPA for long sequences.


## Résultats
- val_bpb: 1.094853
- peak_vram_mb: 15735.5
- status: discarded
