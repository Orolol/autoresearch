Even more extreme non-uniform MLP widths — 1.25x/4.75x split instead of 1.5x/4.5x, iso-compute with baseline average of 3x.

## What changed

One-line change in `train.py` Block.__init__ (line 141):
- `expansion = 1.5 if layer_idx < half else 4.5` -> `expansion = 1.25 if layer_idx < half else 4.75`

Layer-by-layer MLP hidden dims:
- Layers 0-3: 1.25x = 640 hidden (was 768)
- Layers 4-7: 4.75x = 2432 hidden (was 2304)

Total MLP params: 4 * (512*640*2) + 4 * (512*2432*2) = 2,621,440 + 9,961,472 = 12,582,912. Identical to 8 * (512*1536*2) = 12,582,912 (baseline uniform 3x). Zero net parameter change.

## Why

- **Winning trend**: Non-uniform MLP is the clearest pattern: 2x/4x (exp122, -0.00114) -> 1.5x/4.5x (exp129, -0.00109). Each step pushing more compute into late layers improved quality.
- **Late layers need more**: Layers 4-7 receive residual + U-Net skip info and shape features for lm_head. Wider MLPs give them more capacity.
- **Early layers can afford less**: Their outputs are available to late layers through U-Net skip connections.
- **Safe dims**: 640 = 5x128 (above the 512 "too narrow" crash in exp123), 2432 = 19x128. Both tensor-core friendly.

## Risks

- **Neutral (medium)**: 1.5x/4.5x may already be the sweet spot, with diminishing returns from further asymmetry.
- **Regression (low)**: 1.25x=640 might start bottlenecking early layers.
- **Crash (none)**: Pure dimension change, all dims safe.
- Expected range: -0.003 to +0.003

## If it works

- Push to 1.0x/5.0x (but 1x=512=model_dim, same as exp123 crash territory — may need careful handling)
- Try 3-tier non-uniform: [1.0, 1.25, 1.5, 2.5, 3.5, 4.5, 5.0, 5.0] or similar
- Non-uniform MLP is a clear winning direction for this architecture

## If it fails

- 1.5x/4.5x (exp129) is the sweet spot for MLP asymmetry. Further pushing is dead.
- Try intermediate RoPE range (1500, 8000, 60000, 300000) — explicitly suggested in prompt.md, untested
- Consider FP8 retry with different approach (MLP-only, simpler torch API)
- DeltaNet/GLA remains the only untried high-potential architectural change


## Résultats
- val_bpb: 1.068039
- peak_vram_mb: 11985.0
- status: discarded
