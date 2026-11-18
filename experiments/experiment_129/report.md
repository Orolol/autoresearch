More extreme non-uniform MLP widths — 1.5x/4.5x split instead of 2x/4x, iso-compute with baseline average of 3x.

## What changed

One-line change in `train.py` Block.__init__ (line 141):
- `expansion = 2 if layer_idx < half else 4` -> `expansion = 1.5 if layer_idx < half else 4.5`

Plus a safety cast in MLP.__init__ (line 123):
- `hidden = expansion * config.n_embd` -> `hidden = int(expansion * config.n_embd)`

Layer-by-layer MLP hidden dims:
- Layers 0-3: 1.5x = 768 hidden (was 1024)
- Layers 4-7: 4.5x = 2304 hidden (was 2048)

Total MLP params: 4 * (512*768*2) + 4 * (512*2304*2) = 3,145K + 9,437K = 12,583K. Identical to 8 * (512*1536*2) = 12,583K (baseline uniform 3x) and to previous 2x/4x. Zero net parameter change.

## Why

- **Direct follow-up on exp122** (-0.00114, new best): non-uniform MLP works. More extreme asymmetry tests how far the "late layers need more" principle extends.
- **Late layers are prediction-critical**: layers 4-7 receive combined residual + U-Net skip information and directly shape features for lm_head. Wider MLPs give them more capacity to exploit this rich input.
- **Early layers can afford less**: layers 0-3 do basic feature extraction. Their outputs still contribute to final prediction via U-Net skip connections, so reduced capacity is compensated.
- **Iso-compute**: same total params and FLOPs. Any improvement is pure architectural benefit.
- **Token shift exhausted**: K+V works (exp126), Q hurts (exp127), V-only hurts (exp128). Need a new direction.

## Risks

- **Regression (medium)**: 1.5x=768 in early layers may be too narrow. MLP 2.5x uniform was +0.003 (exp15). But that was ALL layers at 2.5x; here only 4 layers are at 1.5x.
- **Neutral (medium)**: 2x/4x may already be the sweet spot. The original -0.00114 was within noise (~0.003 variance).
- **Crash (none)**: Just different linear dimensions, all multiples of 128.
- Expected range: -0.003 to +0.005

## If it works

- Push even further: 1x/5x (if not too narrow) or 1.5x/5x (non-iso, but more capacity)
- Try asymmetric 3-tier: [1.5, 1.5, 3, 3, 3, 4.5, 4.5, 4.5]
- Combine with other untested ideas (intermediate RoPE range)

## If it fails

- 2x/4x is the sweet spot for non-uniform MLP. Further asymmetry is dead.
- Non-uniform MLP exploration complete. Focus on:
  - FP8 retry with different approach (MLP-only, or wait for torchao)
  - DeltaNet/GLA (requires significant engineering effort)
  - Intermediate RoPE range (1500, 8000, 60000, 300000) as last hyperparameter idea


## Résultats
- val_bpb: 1.067071
- peak_vram_mb: 11984.2
- status: IMPROVED
