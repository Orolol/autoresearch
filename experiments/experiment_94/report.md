DEPTH=9 (6th attempt) with U-Net skip bug fix for odd depths — previous 5 attempts all crashed.

## What changed

Three changes in train.py:

1. **Bug fix** (forward, line ~324): `if i >= half:` changed to `if i >= half and mirror < half:`. With odd n_layer (e.g. 9), the middle layer (i=4) computes `mirror = 9-1-4 = 4`, then tries `skip_lambdas[4]` which doesn't exist (size n_layer//2 = 4, indices 0-3). exp88 crashed from this bug. exp92/93 had the fix but crashed on infra.

2. `DEPTH = 9` (was 8)
3. `ASPECT_RATIO = 56` (was 64, to keep model_dim=512 after rounding: 9*56=504 -> ceil to 512)

Result: 9 transformer layers at 512-dim with 4 heads. Everything auto-adapts:
- U-Net skips: 4 early layers (0-3) save, 4 late layers (5-8) receive, layer 4 is middle (no skip)
- VE on layers 0 and 8 (shared weights, separate gates)
- Multi-scale RoPE: 4 bases for 4 heads, unchanged
- Window pattern SSSL cycles: S,S,S,L,S,S,S,L,L (last forced to L)

## Why

Depth is the single most valuable axis (exp18 DEPTH=7: +0.033 catastrophe). DEPTH=9 with the correct bug fix has never successfully run on GPU despite 5 attempts. exp88 had the IndexError bug; exp92/93 had the fix but crashed on RunPod infra.

Back-of-envelope: current best 1.072821 (8 layers) minus depth benefit from accumulated improvements (multi-scale RoPE, U-Net skips, VE bookends all specifically help deeper models) suggests ~1.070-1.071.

## Risks

- ~8-10% fewer training steps (~750 vs ~830) due to extra layer compute
- If step reduction outweighs capacity gain: regression ~0.001-0.003
- Zero OOM risk (standard Block addition)
- Bug fix is proven correct (exp92 code review)
- Main risk is infra crash again (RunPod)

## If it works

- Try DEPTH=10 (push depth further)
- Try reducing MLP from 3x to 2.75x to recover some steps at 9 layers
- Try VE on middle layer (4) in addition to bookend layers (0, 8)

## If it fails

- 8 layers is truly optimal for 300s budget
- Focus on non-depth changes: DeltaNet/GLA, MoE MLP, Schedule-Free Adam
