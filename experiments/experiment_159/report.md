Hybrid local/global attention heads in middle layers — 1 global head + 3 local heads for cross-block context without full compute.

## What changed

In `CausalSelfAttention.forward` (train.py), the `elif self.local_attn:` branch (layers 2-5):
- Previously: all 4 heads used block-diagonal local attention (block_size=256)
- Now: last head (index 3, RoPE base=160000) uses full causal SDPA, first 3 heads (indices 0-2, RoPE bases 2500/10000/40000) use block-diagonal local attention
- Implementation: split Q/K/V by head index, run two separate SDPA calls (one global, one local), concatenate results preserving head ordering
- All shapes are static (n_global=1, n_local=3 are constants), torch.compile compatible

Total new parameters: 0. Compute change: middle layer attention FLOPs increase from ~12.5% to ~43% of full attention FLOPs (3 local + 1 global vs 4 local). Estimated ~5-8% fewer total training steps.

## Why

Middle layers (2-5) use block-diagonal local attention with block_size=256. Every head can only see tokens within its 256-token block. Cross-block information comes only from:
1. Earlier full-attention layers (0, 1) via residual stream
2. Token shift (1-token lookback for K/V)
3. U-Net skip connections from mirror layers

This means middle layers are blind to long-range patterns. The last head has RoPE base=160000 (effective range ~1600 tokens), but it's confined to a 256-token block — its positional encoding is wasted.

By giving this head full causal attention:
- Each middle layer gets a "global oracle" summarizing the full sequence
- The long-range RoPE head can actually USE its long-range positional encoding
- 3 local heads still provide efficient fine-grained local processing
- The model can route global vs local information through different heads

This is different from exp150 (extending local attention to more layers) which tested WHICH layers are local. This tests WITHIN-LAYER hybrid: some heads local, some global.

## Risks

- **Regression (medium)**: More compute per step → ~5-8% fewer training steps. Exp149 gained -0.0015 from local attention, partly from more steps. Giving back some of that speed may negate the benefit.
- **Neutral (medium)**: Full-attention layers 0,1,6,7 may already provide sufficient global context to middle layers via the residual stream.
- **Crash (very low)**: Standard SDPA calls with static shapes. Two separate SDPA calls per local layer instead of one.
- **OOM (none)**: No additional memory beyond one head of full attention per middle layer.

Expected range: -0.003 to +0.003.

## If it works

- Try n_global=2 (2 global + 2 local) for more global context at higher compute cost
- Try different head assignment: make the 2 longest-range heads global (bases 40000, 160000)
- Try hybrid only in layers 3-4 (deepest middle layers) while keeping 2,5 fully local
- Consider if this enables reducing full-attention bookend layers (e.g., layers 1,6 could become fully local)

## If it fails

- Full-attention bookend layers provide sufficient global context; middle layer locality is fine
- The compute cost of 1 global head outweighs the information benefit
- The model is near its architecture ceiling at ~830 steps
- Next directions: FP8 matmuls (biggest "more steps" lever), DeltaNet/GLA (full attention replacement)


## Résultats
- val_bpb: 1.064835
- peak_vram_mb: 12759.3
- status: discarded
