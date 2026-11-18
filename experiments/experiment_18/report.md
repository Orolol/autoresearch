Reduce DEPTH from 8 to 7 layers to decrease compute per step and get more training steps in 5 minutes.

## What changed
- `DEPTH = 8` -> `DEPTH = 7` (single line change)
- Model dimension stays 512 (7*64=448, rounded up to nearest HEAD_DIM=128 = 512)
- 4 heads, same MLP 3x (1536 hidden), same everything else
- Removes ~2.6M transformer matrix params (~12.5% of transformer compute)

## Why
- **"More steps" is the ONLY strategy that has ever beaten baseline.** Exp14 (MLP 4x->3x) gave -0.008, the biggest single improvement ever. The model is severely undertrained in 5 minutes — every step saved translates to an extra optimizer step and better convergence.
- **Removing 1 layer is the next natural step** in the "less compute per step" strategy. MLP 3x reduced width; now we reduce depth. Each layer removed saves ~2.6M matrix params (attention Q/K/V/O + MLP fc/proj).
- **7 layers should provide sufficient capacity**: the model operates at 512 dim with 3x MLP, which gives plenty of representational power per layer. Modern small models (e.g., TinyLlama) operate with similar depth-to-width ratios.
- **No quality mechanism removed**: same attention, same MLP, same VE (last layer only), same x0_lambdas, same z-loss, same softcap. Just one fewer sequential transformation.
- **VE stays on last layer**: `has_ve(layer_idx, n_layer)` returns True for layer 6 (last of 7), so VE is still active.

## Expected outcome
- If reduced compute gives enough extra steps to offset depth loss: improvement of ~0.001-0.005
- If depth loss exactly matches step gain: neutral within noise (~0.003)
- If 7 layers is too shallow for good representations: regression of ~0.002-0.005
- No crash, OOM, or instability risk — pure depth change


## Résultats
- val_bpb: 1.112418
- peak_vram_mb: 9210.2
- status: discarded
