Partitioned Hyperconnections — retry of exp174 (INFRA crash, code never ran). 2-lane residual stream with learned per-sublayer write weights (modded-nanogpt PR#230).

## What changed

Replaced the single residual stream with a 2-lane architecture in `GPT.forward`:

1. **New parameters** (`GPT.__init__`): `lane_attn_write` and `lane_mlp_write`, each `(n_layer, 2)` = 32 total scalars. Initialized to 1.0 (neutral — both lanes identical at start).

2. **Modified forward pass**: Instead of one `x` tensor, maintain `s0` (lane0) and `s1` (lane1):
   - Both lanes receive `resid_lambdas` and `x0_lambdas` scaling identically
   - Both lanes get U-Net skip connections (per-lane skip states)
   - **Attention** reads from `norm(s0)`, writes output to both lanes: `s0 += w_a0 * attn_out`, `s1 += w_a1 * attn_out`
   - **MLP** reads from `norm(s1)`, writes output to both lanes: `s0 += w_m0 * mlp_out`, `s1 += w_m1 * mlp_out`
   - Final output: `x = (s0 + s1) * 0.5` then norm

3. **Optimizer**: Lane params use AdamW with scalar_lr=0.5, betas=(0.96, 0.95).

Total new parameters: 32 scalars. No new FLOPs. ~128MB extra activation memory.

## Why

1. **Exp174 crashed on INFRA** (RunPod deployment), code never tested. This is a direct retry.
2. **modded-nanogpt PR#230** (Feb 2026): -45 steps improvement with 2-lane Partitioned Hyperconnections.
3. **Different from parallel attn+MLP** (exp60, +0.034): that failed because both sublayers shared the same input. Here, each sublayer reads from its own dedicated lane while writing to both.
4. **Neutral at init**: write weights = 1.0 means both lanes are identical at step 0, so the model starts equivalent to the standard single-stream baseline.
5. **Forced structural change**: Last 3+ experiments before exp174 were HP tweaks.

## Risks

- **Crash (low)**: No custom ops, no dynamic shapes. Scalar-tensor multiplications compatible with torch.compile(dynamic=False).
- **Regression (medium)**: If write weights don't differentiate in ~830 steps, the model wastes capacity on two nearly-identical lanes.
- **OOM (none)**: 2nd lane adds ~128MB, well within 32GB headroom (current peak: 12.2GB).
