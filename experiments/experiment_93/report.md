DEPTH=9 retry (5th attempt) with U-Net skip bug fix for odd depths — exp86/87/88/92 all crashed.

## What changed

Three changes in train.py:

1. **Bug fix** (line 325): `if i >= half:` changed to `if i >= half and mirror < half:`. With odd n_layer (e.g. 9), the middle layer (i=4) computes `mirror = 9-1-4 = 4`, then tries `skip_lambdas[4]` which doesn't exist (size n_layer//2 = 4, indices 0-3). This caused exp86/87 crashes. Exp92 had this fix but crashed on infra.

2. `DEPTH = 9` (was 8)
3. `ASPECT_RATIO = 56` (was 64, to keep model_dim=512 after rounding: 9*56=504 -> ceil to 512)

Result: 9 transformer layers at 512-dim with 4 heads. Everything auto-adapts:
- U-Net skips: 4 early layers (0-3) save, 4 late layers (5-8) receive, layer 4 is middle (no skip)
- VE on layers 0 and 8 (shared weights, separate gates)
- Multi-scale RoPE: 4 bases for 4 heads, unchanged
- Window pattern SSSL cycles: S,S,S,L,S,S,S,L,L (last forced to L)

## Why

Depth is the single most valuable axis (exp18 DEPTH=7: +0.033 catastrophe). Exp22 tested DEPTH=9 before all gradient flow improvements and got 1.080504 vs 1.079602 baseline with MLP 2.5x. Since then, accumulated improvements total ~-0.007 (multi-scale RoPE -0.002, U-Net skips -0.001, VE bookends -0.004), and MLP is now 3x (vs 2.5x, known -0.003 better). Back-of-envelope: ~1.070-1.071.

## Risks

- ~8-10% fewer training steps (~750 vs ~830) due to extra layer compute
- If step reduction outweighs capacity gain: regression ~0.001-0.003
- Zero OOM risk (same architecture, one more standard Block)
- Previous 4 crashes were infra (RunPod), not code bugs

## If it works

- Try DEPTH=10 (push depth further)
- Try reducing MLP from 3x to 2.75x to recover some steps at 9 layers
- Try VE on middle layer (4) in addition to bookend layers (0, 8)

## If it fails

- 8 layers is truly optimal for 300s budget
- Focus on non-depth architectural changes: DeltaNet, GLA, or Mixture of Experts MLP
