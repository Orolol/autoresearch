Block-diagonal local attention for middle layers (2-5) with block_size=256, keeping full attention for layers 0, 1, 6, 7.

## What changed

In `CausalSelfAttention.__init__` (train.py):
- Added `self.local_attn` flag: True for layers 2-5, False for layers 0, 1, 6, 7
- Added `self.block_size = 256`

In `CausalSelfAttention.forward` (train.py):
- New `elif self.local_attn` branch in the SDPA section
- Reshapes (B, T, H, D) into (B*n_blocks, block_size, H, D) before SDPA
- Runs standard SDPA with is_causal=True on 256-token blocks
- Reshapes back to (B, T, H, D) after SDPA
- Full attention path unchanged for non-local layers

Zero new parameters. Zero new memory. Pure compute reduction.

## Why

WINDOW_PATTERN="SSSL" was designed to give some layers short attention windows, but SDPA on this GPU path (non-Hopper) ignores window_size entirely — all layers compute full causal attention. Block-diagonal reshaping is a compile-friendly way to actually implement local attention.

The biggest single win in the project's history was MLP 3x (exp14, -0.008) from reduced compute → more training steps. This change exploits the same lever:
- 4 out of 8 layers get ~8x cheaper attention (256^2 vs 2048^2)
- Attention is ~26% of total compute → ~50% attention savings → ~13% overall speedup
- ~100 more training steps in the 5-minute budget

The hybrid design (full attention for layers 0, 1, 6, 7 + local for 2-5) mirrors Longformer/BigBird: global layers handle long-range dependencies while local layers do efficient local refinement. The full-attention layers include both VE bookend layers (0, 7) and their neighbors (1, 6), preserving the most critical global context paths. U-Net skip connections propagate early-layer global features to late local layers.

## Risks

- **Regression (medium)**: Hard block boundaries create blind spots — position 256 can't attend to position 255. Mitigated by: (a) token shift provides 1-token lookback, (b) full-attention layers 0,1 provide global context before local layers, (c) U-Net skips propagate global features to late layers.
- **Neutral (medium)**: If attention compute is a small fraction of total time (MLP dominates), the ~13% speedup may not yield enough extra steps to matter.
- **Crash (low)**: All static shapes, standard SDPA ops. View/reshape on contiguous tensors after RoPE.
- **Torch.compile (low)**: self.local_attn is a constant attribute per module instance — should trace correctly.
- Expected range: -0.005 to +0.010

## If it works

- Try block_size=128 (16x cheaper attention, more steps but narrower context)
- Try block_size=512 (4x cheaper, wider context, less speedup)
- Try making only 2 middle layers local (3-4) instead of 4 (2-5)
- Apply to more layers (keep only layer 0 and 7 as full attention)

## If it fails

- Block boundaries too harmful → try overlapping blocks or sliding-window via manual attention mask
- Speedup too small → the only remaining "more steps" lever is FP8 matmuls
- Move to DeltaNet/GLA as the last bold architectural bet


## Résultats
- val_bpb: 1.063080
- peak_vram_mb: 12499.2
- status: IMPROVED
