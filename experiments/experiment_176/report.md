Partitioned Hyperconnections — 2-lane residual stream with learned per-sublayer write weights (modded-nanogpt PR#230, 3rd attempt after 2 INFRA crashes).

## What changed

In `train.py`, replaced the single residual stream with a 2-lane architecture:

1. **New parameter** (`GPT.__init__`): `post_lambdas` of shape `(n_layer, 2, 2)` = 32 scalars. `post_lambdas[i, s, l]` controls how sublayer `s` (0=attn, 1=mlp) at layer `i` writes to lane `l` (0 or 1). Initialized to 1.0 (neutral).

2. **Modified forward pass** (`GPT.forward`): Instead of one `x` tensor, maintain `s0` (lane0) and `s1` (lane1):
   - Both lanes receive `resid_lambdas` and `x0_lambdas` scaling identically
   - Both lanes get independent U-Net skip connections (`skip_s0`, `skip_s1`)
   - **Attention** reads from `norm(s0)`, writes to both: `s0 += post[i,0,0] * attn_out`, `s1 += post[i,0,1] * attn_out`
   - **MLP** reads from `norm(s1)`, writes to both: `s0 += post[i,1,0] * mlp_out`, `s1 += post[i,1,1] * mlp_out`
   - Final output: `x = (s0 + s1) * 0.5` then norm

3. **Optimizer**: post_lambdas use AdamW with scalar_lr=0.5, betas=(0.96, 0.95) — same as other per-layer scalars.

4. **Block.forward** is bypassed — `block.attn()` and `block.mlp()` are called directly from `GPT.forward`.

Total new parameters: 32 scalars. No new FLOPs. ~128MB extra activation memory.

## Why

1. **Exp174/175 both crashed on INFRA** (RunPod template conflict), code was never GPU-tested. This is the 3rd attempt with the infra bug fixed.
2. **modded-nanogpt PR#230** (Feb 2026): -45 steps improvement with 2-lane Partitioned Hyperconnections on GPT-2 124M.
3. **Different from parallel attn+MLP** (exp60, +0.034): that failed because both sublayers shared the same input. Here, each sublayer reads from its own dedicated lane while writing to both.
4. **Neutral at init**: post_lambdas = 1.0 means both lanes are identical at step 0, mathematically equivalent to the standard single-stream baseline.
5. **Forced structural change**: exp171-173 were all hyperparameter tweaks.

## Risks

- **Crash (low)**: No custom ops, no dynamic shapes. Scalar-tensor multiplications compatible with torch.compile(dynamic=False).
- **Regression (medium)**: If post_lambdas don't learn useful differentiation in ~830 steps, the lanes remain near-identical and the (s0+s1)/2 merge slightly smooths the output.
- **OOM (none)**: 2nd lane adds ~128MB, well within 32GB headroom (current peak: 12.2GB).

## If it works

- Tune post_lambdas init (e.g., asymmetric [1.0, 0.5] to force faster differentiation)
- Try 3 or 4 lanes
- Combine with lane-specific learning rates or lane-specific depth

## If it fails

- Try asymmetric initialization to break symmetry faster
- Move to DeltaNet/GLA hybrid (priority #1, highest potential but most complex)
- The model may be at its practical ceiling (~1.062) with standard softmax attention
