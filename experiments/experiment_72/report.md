Enable actual sliding window attention via flex_attention for the SDPA path — the SSSL window pattern was designed but never functional on RTX 5090.

## What changed

Used `torch.nn.attention.flex_attention` with `create_block_mask` to implement real sliding window attention on non-Hopper GPUs. The SSSL window pattern (3 short + 1 long, repeated) was always in the code but SDPA ignored the window_size parameter — all 8 layers did full 2048 causal attention.

With this change:
- 6/8 layers ("S" layers) use block-sparse sliding window attention (window=1024) via flex_attention
- 2/8 layers ("L" layers, including forced-last) keep full causal attention via SDPA
- Block masks are precomputed in `init_weights()` and stored per-layer on `CausalSelfAttention._block_mask`
- Graceful fallback: if flex_attention import or mask creation fails, all layers silently use SDPA

~25 lines changed across imports, CausalSelfAttention, and GPT.init_weights.

## Why

The "more training steps" thesis is the strongest proven lever in this series (MLP 3x gave -0.008 by being faster). Sliding window attention skips ~50% of attention compute on 6/8 layers. If attention is ~30% of total compute, this gives ~10-15% overall speedup → ~80-130 more training steps in the 5-minute budget.

flex_attention (PyTorch 2.5+) generates compiled Triton kernels that skip entire blocks of masked attention, giving real compute savings (unlike SDPA with dense attn_mask which still computes masked entries).

Additionally, the sliding window inductive bias (local attention for most layers, global for every 4th + last) was part of the original architecture design and may also improve quality.

## Risks

- **Triton kernel speed**: flex_attention's Triton kernels may be slower per-FLOP than SDPA's cuDNN/FlashAttention backends. If flex_attention is 2x slower per FLOP, the window savings (50% fewer FLOPs) cancel out and we get no benefit. If flex_attention is comparable per-FLOP, we get ~10-15% net speedup.
- **torch.compile compatibility**: BlockMask objects stored as module attributes may confuse the compiler. Mitigated by: each layer has a fixed _block_mask (constant at trace time), and flex_attention is designed for torch.compile.
- **Quality regression**: Full attention on all layers may actually be better than windowed. The last layer and every 4th layer still have full attention, which should preserve long-range dependency modeling. Mild regression risk ~0.001-0.003.
- **Compilation overhead**: flex_attention requires Triton kernel compilation during warmup. This increases warmup time but doesn't count toward the 5-minute budget (steps 0-10 excluded).

## If it works

- Try shorter window: WINDOW_PATTERN "SSSL" with short_window = sequence_len // 4 = 512 for even more speedup
- Try different patterns: "SSL" (more global layers) or "SSSSL" (more local layers)
- Combine with z-loss 2e-4 (clear directional signal: more z-loss helps)
- Try DEPTH=9 with sliding window (the speed gain could compensate for the extra layer's compute)

## If it fails

- If crash: likely torch.compile + BlockMask incompatibility. Try creating masks as global variables instead of module attributes.
- If regression: sliding window hurts quality. Full attention is genuinely better for all layers at this scale.
- If neutral (no speedup): flex_attention's Triton kernels are too slow to benefit from window masking. Try FP8 matmuls instead (different speed lever).
- Fall back to z-loss 2e-4, or try simple 2-expert MoE in MLP layers.
