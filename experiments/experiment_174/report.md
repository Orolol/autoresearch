Partitioned Hyperconnections — 2-lane residual stream with learned per-sublayer write weights (modded-nanogpt PR#230).

## What changed

In `train.py`, replaced the single residual stream with a 2-lane architecture:

1. **New parameters** (`GPT.__init__`): `lane_attn_write` and `lane_mlp_write`, each `(n_layer, 2)` = 32 total scalars. Initialized to 1.0 (neutral).

2. **Modified forward pass** (`GPT.forward`): Instead of one `x` tensor flowing through blocks, maintain `s0` and `s1`:
   - Both lanes get `resid_lambdas` and `x0_lambdas` scaling identically
   - Both lanes get U-Net skip connections (per-lane skip states)
   - **Attention** reads from `norm(s0)`, writes output to both lanes: `s0 += w_a0 * attn_out`, `s1 += w_a1 * attn_out`
   - **MLP** reads from `norm(s1)`, writes output to both lanes: `s0 += w_m0 * mlp_out`, `s1 += w_m1 * mlp_out`
   - Final output: `x = (s0 + s1) * 0.5` then norm

3. **Optimizer**: Lane params use AdamW with scalar_lr=0.5, betas=(0.96, 0.95).

4. **Block.forward** is no longer called — `block.attn()` and `block.mlp()` are called directly from `GPT.forward`.

Total new parameters: 32 scalars. No new FLOPs. ~128MB extra activation memory.

## Why

Three motivations:

1. **Forced architectural change**: Last 3 experiments (171-173) were all hyperparameter tweaks. The protocol requires a structural change.

2. **modded-nanogpt PR#230** (Feb 2026): Partitioned Hyperconnections with 2 lanes showed -45 steps improvement in the GPT-2 speedrun. The paper (Hyperconnections, 2024) shows that multi-lane residual streams improve gradient flow by giving sublayers dedicated read/write paths.

3. **Different from parallel attn+MLP** (exp60, +0.034): Parallel attn+MLP failed because both sublayers shared the same input and couldn't build on each other. Partitioned Hyperconnections maintain sequential dependencies (MLP sees attention's contribution via lane1) while allowing richer residual routing.

**Key property**: At initialization (all write weights = 1.0), both lanes are identical at every step, so the model output exactly matches the standard single-stream baseline. The lanes only differentiate as write weights learn different values during training.

## Risks

- **Crash (low)**: No custom ops, no dynamic shapes. Just scalar-tensor multiplications fully compatible with torch.compile(dynamic=False).
- **Regression (medium)**: If write weights don't learn useful differentiation in ~830 steps, the model wastes capacity maintaining two nearly-identical lanes. The (s0+s1)/2 merge is an averaging operation that could slightly smooth gradients.
- **OOM (none)**: 2nd lane adds ~128MB. Current peak VRAM is 12.2GB out of 32GB.
- **Compilation (low)**: Two skip_states lists instead of one may increase compile time slightly.

## If it works

- Tune the init values (e.g., asymmetric init to break symmetry faster)
- Try 3 or 4 lanes
- Combine with lane-specific learning rates

## If it fails

- Try asymmetric initialization (e.g., lane_attn_write init = [1.0, 0.5]) to force differentiation
- The lanes may need more steps to differentiate — consider with batch size reduction for more steps
- Move to DeltaNet/GLA (priority #1, most complex but highest potential)
