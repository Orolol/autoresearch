DEPTH=9 with U-Net skip bug fix for odd depths — exp86/87 crashed due to IndexError in skip_lambdas indexing, not infra.

## What changed

Three changes in train.py:

1. **Bug fix** (line 325): `if i >= half:` changed to `if i >= half and mirror < half:`. With odd n_layer (e.g. 9), the middle layer (i=4) computes `mirror = 9-1-4 = 4`, then tries `skip_lambdas[4]` which doesn't exist (size n_layer//2 = 4, indices 0-3). This was the real cause of exp86/87 crashes, not RunPod infra.

2. `DEPTH = 9` (was 8)
3. `ASPECT_RATIO = 56` (was 64, to keep model_dim=512 after rounding: 9*56=504 -> ceil to 512)

Result: 9 transformer layers at 512-dim with 4 heads. Everything auto-adapts:
- U-Net skips: 4 early layers (0-3) save, 4 late layers (5-8) receive, layer 4 is middle (no skip)
- VE on layers 0 and 8 (shared weights, separate gates)
- Multi-scale RoPE: 4 bases for 4 heads, unchanged
- Window pattern SSSL cycles: S,S,S,L,S,S,S,L,L (last forced to L)

## Why

Depth is the single most valuable axis (exp18 DEPTH=7: +0.033 catastrophe). Exp22 tested DEPTH=9 before all gradient flow improvements and got 1.080504 vs 1.079602 baseline (+0.001). But exp22 used MLP 2.5x (known -0.003 worse than 3x). Correcting for this: DEPTH=9 with 3x MLP would have been ~1.078, which would have beaten that era's baseline.

Since exp22, accumulated improvements total ~-0.007: multi-scale RoPE (-0.002), U-Net skips (-0.001), VE bookends (-0.004). All of these specifically help deeper models. Back-of-envelope: 1.072821 (current best with 8 layers) - depth benefit ≈ ~1.070-1.071.

## Risks

- Bug fix is safe: condition `mirror < half` correctly skips middle layer for odd depths, no effect on even depths
- ~8-10% fewer training steps (~750 vs ~830) due to extra layer compute
- If step reduction outweighs capacity gain: mild regression ~0.001-0.003
- Zero OOM risk (standard Block addition, no new parameters beyond one extra layer)

## If it works

- Try DEPTH=10 (push depth further)
- Try reducing MLP from 3x to 2.75x to recover some steps at 9 layers
- Try wider RoPE base range (1000, 10000, 100000, 1000000)
- Try VE on middle layer (4) in addition to bookend layers (0, 8)

## If it fails

- 8 layers is optimal for 300s budget even with improved gradient flow
- Try wider multi-scale RoPE range (1000, 5000, 50000, 500000)
- Try DeltaNet or GLA (replace softmax attention entirely)
- Try per-sublayer output scaling (attn_alpha, mlp_alpha per layer)
