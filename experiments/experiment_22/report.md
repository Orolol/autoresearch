Iso-compute depth-for-width swap: DEPTH=9 + MLP 2.5x + ASPECT_RATIO=56 to add an extra transformer layer at near-zero compute cost.

## What changed
- `DEPTH = 8` → `DEPTH = 9` (one extra transformer layer)
- `ASPECT_RATIO = 64` → `ASPECT_RATIO = 56` (keeps model dim at 512: 9×56=504, rounds to 512)
- MLP hidden dim: `3 * n_embd` → `int(2.5 * n_embd)` (1536 → 1280)
- 3 lines changed

## Why
- **DEPTH is the most critical factor.** Exp18 (DEPTH=7) was catastrophic (+0.033 regression). This means each layer carries enormous value for model quality at this scale.
- **MLP 2.5x is only slightly worse than 3x.** Exp15 (2.5x alone) gave 1.082 vs 1.079 for 3x — only +0.003. The quality cost is small.
- **Iso-compute swap**: 8 layers × (4×512² + 2×512×1536) = 20.97M matrix params vs 9 layers × (4×512² + 2×512×1280) = 21.23M matrix params (+1.25%). Nearly identical total compute, so step count barely changes.
- **The depth benefit should dwarf the MLP cost.** Even with strong diminishing returns from DEPTH 8→9, the benefit should be +0.005-0.010. The MLP narrowing costs only ~0.003. Net expected improvement: +0.002-0.007.
- **Model dim stays 512** (9×56=504, rounded up to 512 with HEAD_DIM=128). Same number of heads (4), same attention compute per layer.
- **VE stays on last layer** (layer 8 of 9). All other mechanisms unchanged (z-loss, softcap=15, x0_lambdas, QK-norm).

## Key insight
The asymmetry between depth and width at this scale: losing a layer costs +0.033, but shrinking MLP from 3x to 2.5x costs only +0.003. Trading MLP width for depth is a favorable exchange.

## Expected outcome
- If depth benefit (diminishing returns from 8→9) exceeds MLP cost: improvement of ~0.002-0.007
- If depth benefit and MLP cost exactly cancel: neutral within noise (~0.003)
- If 9 layers is too many steps per iteration and MLP 2.5x capacity is insufficient: regression of ~0.002-0.005
- No crash, OOM, or instability risk — pure architecture reshaping
- Attention compute increases 12.5% (9 vs 8 layers), MLP compute decreases 17% per layer. Net compute per step: roughly neutral.


## Résultats
- val_bpb: 1.080504
- peak_vram_mb: 10855.2
- status: discarded
