Continuous MLP width ramp [1.0, 1.25, 1.75, 2.25, 3.25, 4.25, 5.0, 5.25] replacing coarse 1.5x/4.5x two-step split for iso-compute smooth per-layer capacity allocation.

## What changed

In `train.py`, replaced the 2-step MLP expansion in `Block.__init__`:

- Old: `expansion = 1.5 if layer_idx < half else 4.5` (layers 0-3 get 1.5x, layers 4-7 get 4.5x)
- New: `_MLP_RAMP = [1.0, 1.25, 1.75, 2.25, 3.25, 4.25, 5.0, 5.25]` — per-layer smooth ramp

Hidden dimensions per layer (n_embd=512):
- Layer 0: 512, Layer 1: 640, Layer 2: 896, Layer 3: 1152
- Layer 4: 1664, Layer 5: 2176, Layer 6: 2560, Layer 7: 2688

Average expansion: 3.0x = iso-compute with baseline.

## Why

1. **Priority queue #1 pivot**: After 4 consecutive chunk attention failures (exp194-197), the priority queue explicitly recommends this as the first alternative axis.
2. **Never GPU-tested**: Exp123 (ramp [1,2,2,3,3,4,4,5]) and exp124 (ramp [2,2,2,3,3,3,4,5]) both INFRA-crashed. No ramp has ever been tested on GPU.
3. **Strong non-uniform MLP trend**: uniform 3x → 2x/4x (exp122, -0.002) → 1.5x/4.5x (exp129, -0.001). Each refinement improved. The trend suggests finer granularity could help.
4. **Information flow argument**: Early layers primarily do token mixing via attention; they need less MLP capacity. Late layers aggregate and transform for prediction; they need maximum MLP. A smooth gradient matches this intuition better than a step function.
5. **Clean AB test**: One conceptual change (MLP distribution shape), iso-compute, no new parameters, no new ops.

## Risks

- **Regression (low-medium)**: Layer 0 gets only 1.0x MLP (512 hidden) — if it needs more capacity for initial feature extraction, this bottlenecks. But exp130's 1.25x/4.75x was only +0.001 vs 1.5x/4.5x, suggesting early layers tolerate narrow MLPs.
- **Crash (very low)**: Standard nn.Linear with different hidden dims. Static shapes, compile-friendly.
- **Muon fragmentation (low)**: 16 unique MLP weight shapes (8 for c_fc, 8 for c_proj) instead of 4. More Muon groups but each is still a valid matrix for orthogonalization.
- **Neutral (medium)**: If the 2-step split was already near-optimal for this 8-layer architecture, smooth ramp adds no value.


## Résultats (after crash fix)
- val_bpb: 1.051193
- peak_vram_mb: 14131.5
- status: discarded
