Remove all Value Embeddings (VE=0) to test if the model can match baseline without any VE overhead.

## What changed
- `has_ve()` now returns `False` unconditionally (was `layer_idx == n_layer - 1`)
- This removes the last-layer value embedding (32768 * 512 = 16.8M params, ~22% of model)
- All VE-related code paths (ve_gate, value_embeds lookup, ve mixing in attention) become no-ops
- No other changes to architecture or hyperparameters

## Why
- The baseline (1.088007) uses VE on the last layer only, which was already a reduction from VE on all layers
- The notes suggest testing VE=0 as the next step if VE-last-only works
- Removing 16.8M params should meaningfully speed up forward/backward passes
- More steps in 5 minutes has been the dominant factor for convergence
- If quality holds, this is a big simplification win (less code, less memory, faster training)

## Expected outcome
- Slightly faster training (more steps in 5 min)
- val_bpb within ~0.003 of baseline if VE isn't critical
- Possible regression of ~0.005-0.01 if VE provides important signal


## Résultats
- val_bpb: 1.111258
- peak_vram_mb: 11316.0
- status: discarded
