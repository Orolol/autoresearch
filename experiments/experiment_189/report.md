Partitioned Hyperconnections — 2-lane residual stream where attention reads lane 0, MLP reads lane 1, both write to both lanes with learned weights (modded-nanogpt PR#230).

## What changed

In `train.py`, implemented Partitioned Hyperconnections — a 2-lane residual stream architecture inspired by modded-nanogpt PR#230 (reported -45 steps improvement):

1. **New parameters** (`GPT.__init__`):
   - `attn_write`: (n_layer, 2) — learned per-layer write weights for attention output to each lane
   - `mlp_write`: (n_layer, 2) — learned per-layer write weights for MLP output to each lane
   - Total: 8 layers x 2 sublayers x 2 lanes = 32 new scalar parameters (negligible)

2. **Modified forward** (`GPT.forward`):
   - Residual stream split into 2 lanes: `x_a` (lane 0) and `x_m` (lane 1), both initialized from the same `norm(wte(idx))`
   - Attention reads from `norm(x_a)`, writes to both lanes: `x_a += attn_write[i,0] * attn_out`, `x_m += attn_write[i,1] * attn_out`
   - MLP reads from `norm(x_m)`, writes to both lanes: `x_a += mlp_write[i,0] * mlp_out`, `x_m += mlp_write[i,1] * mlp_out`
   - resid_lambdas, x0_lambdas, and U-Net skip connections apply to both lanes independently
   - Final output: `x = (x_a + x_m) * 0.5` then norm
   - Block.forward is bypassed — block.attn and block.mlp called directly from GPT.forward

3. **Init** (`GPT.init_weights`):
   - All write weights initialized to 1.0 → at step 0, both lanes receive identical updates and evolve identically → `(x_a + x_m) / 2 = x_a = x_m = x_current` → exact baseline equivalence

4. **Optimizer** (`GPT.setup_optimizer`):
   - Lane params added as AdamW group with scalar_lr and (0.96, 0.95) betas, matching x0/skip params

## Why

1. **Priority queue directive**: Partitioned Hyperconnections is priority #2. Attempted twice (exp174/175) but both crashed on INFRA — code was never GPU-tested.
2. **modded-nanogpt PR#230 validation**: Reported -45 steps improvement in Feb 2026. The 2-lane residual allows attention and MLP to specialize in what information they receive — attention gets a "clean" residual without MLP noise from prior layers, and MLP gets a residual without attention noise.
3. **Structural, not HP tweak**: Last 3+ experiments (184-186) were small tweaks. The rules require a structural change.
4. **Different from parallel attn+MLP** (exp60, +0.034): That ran both sublayers on the same input in parallel. Here, sublayers are sequential but read from different lanes of the residual stream.
5. **Baseline equivalent at init**: Both lanes start identical and receive identical updates → zero regression risk at step 0. Over training, the write weights differentiate, allowing lane specialization.

## Risks

- **Regression (low-medium)**: The /2 merge could dilute signal if the lanes don't differentiate enough. But init is neutral.
- **Crash (low)**: All ops are standard (scalar multiply, add, clone). torch.compile should handle 2 separate tensor paths fine since shapes are static.
- **Compile time (low-medium)**: The forward graph has ~2x more nodes for the residual stream, but the heavy computation (attention, MLP) is unchanged.
- **Memory (low)**: 2x residual tensors + 2x skip states. ~320MB extra, well within 32GB VRAM budget.
- **Neutral (medium)**: 830 steps may not be enough for the lanes to differentiate meaningfully from the neutral init.


## Résultats
- val_bpb: 1.055929
- peak_vram_mb: 15592.3
- status: discarded
