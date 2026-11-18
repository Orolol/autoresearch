DEPTH=9 (7th attempt) with U-Net skip bug fix for odd depths — previous 6 attempts all crashed (bug or infra).

## What changed

Three changes in train.py:

1. **Bug fix** (forward, line ~325): `if i >= half:` changed to `if i >= half and mirror < half:`. With odd n_layer (e.g. 9), the middle layer (i=4) computes `mirror = 9-1-4 = 4`, then tries `skip_lambdas[4]` which doesn't exist (size n_layer//2 = 4, indices 0-3). This caused the exp86/87/88 crashes. Exp92/93/94 had this fix but crashed on RunPod infra.

2. `DEPTH = 9` (was 8)
3. `ASPECT_RATIO = 56` (was 64, to keep model_dim=512 after rounding: 9*56=504 -> ceil to 512)

Result: 9 transformer layers at 512-dim with 4 heads. Everything auto-adapts:
- U-Net skips: 4 early layers (0-3) save, 4 late layers (5-8) receive, layer 4 is middle (no skip)
- VE on layers 0 and 8 (shared weights, separate gates)
- Multi-scale RoPE: 4 bases for 4 heads, unchanged
- Window pattern SSSL cycles: S,S,S,L,S,S,S,L,L (last forced to L)

## Why

Depth is the single most valuable architectural axis (exp18 DEPTH=7: +0.033 catastrophe). DEPTH=9 has never successfully run on GPU despite 6 attempts (3 had the IndexError bug, 3 had the fix but crashed on infra). Back-of-envelope: current best 1.072821 (8 layers) minus depth benefit suggests ~1.070-1.071.

## Risks

- ~8-10% fewer training steps (~750 vs ~830) due to extra layer compute
- If step reduction outweighs capacity gain: regression ~0.001-0.003
- Zero OOM risk (standard Block addition)
- Bug fix verified correct for both even and odd depths
- Main risk is infra crash again
