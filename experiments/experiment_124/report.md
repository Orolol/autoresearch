Gradual MLP width ramp [2,2,2,3,3,3,4,5] with min 2x — concentrates maximum capacity in the final prediction layer, iso-compute with baseline.

## What changed

Replaced the 2-step MLP width split (layers 0-3: 2x, layers 4-7: 4x) with a gradual ramp:
- `_MLP_EXPANSIONS = [2, 2, 2, 3, 3, 3, 4, 5]`
- Hidden dims: [1024, 1024, 1024, 1536, 1536, 1536, 2048, 2560]
- Sum = 24 = 8 × 3, identical total MLP params/FLOPs to baseline

Changes from current [2,2,2,2,4,4,4,4]:
- Layers 0-2: unchanged (2x)
- Layer 3: 2x → 3x (slight increase)
- Layers 4-5: 4x → 3x (decrease to fund layer 7)
- Layer 6: unchanged (4x)
- Layer 7: 4x → 5x (maximum capacity for final prediction)

## Why

- **Final layer hypothesis**: The last MLP (layer 7) is the final nonlinear transformation before lm_head. It directly shapes the features used for next-token prediction. Giving it 5x expansion (25% more than current 4x) should improve prediction quality.
- **Exp123 crash mitigation**: The previous gradual ramp [1,2,2,3,3,4,4,5] crashed — likely because 1x MLP (hidden=512=model_dim) was too narrow. This ramp keeps min 2x.
- **Follows exp122 success**: Non-uniform MLP (-0.00114) proved the concept. This refines the distribution.
- **Iso-compute**: Any improvement is pure architectural benefit.

## Risks

- Layers 4-5 dropping from 4x to 3x might lose mid-layer capacity
- 5x last layer might have diminishing returns vs 4x
- Zero crash/OOM risk
- Expected range: -0.003 to +0.003

## If it works

- Try even more aggressive: [2,2,2,2,3,3,4,6] = 24 (6x final layer)
- Try combining with UNEMBEDDING_LR increase (piste #24, never tested)

## If it fails

- 2x/4x split (exp122) was the sweet spot — revert
- Try UNEMBEDDING_LR=0.01 (piste #24, lm_head is 22% of model, never tested)
