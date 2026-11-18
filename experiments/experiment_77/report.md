Mid-MLP normalization (NormFormer-style) — add RMSNorm between ReluSquared activation and output projection in MLP.

## What changed

Single line added in `MLP.forward()` (line 120): `x = norm(x)` after `x = F.relu(x).square()` and before `x = self.c_proj(x)`. This normalizes the hidden activations via RMSNorm before they enter the output projection.

## Why

Every winning experiment in the history improved gradient flow:
- Softcap 13: better gradient through tanh (exp45, -0.001)
- Z-loss: logit calibration gradient (exp9, marginal)
- U-Net skips: shorter gradient paths (exp71, -0.001)

The MLP path is the last major unnormalized internal activation. ReluSquared produces sparse activations with potentially large values (squared). As training progresses and c_fc weights grow, these values can become very large, causing:
1. Gradient magnitude variance through c_proj (large activations → large gradients → instability)
2. c_proj having to implicitly learn to compensate for varying input scale across training

Adding RMSNorm ensures c_proj always receives unit-variance inputs, simplifying its learning task and stabilizing gradient flow. This is the NormFormer approach (Shleifer et al., 2021) — adding norms within sublayers, not just before them.

Zero extra parameters, negligible compute cost (one RMSNorm per layer vs much larger matmuls).

## Risks

- Zero crash/OOM risk (norm() used throughout model, well-supported by torch.compile)
- If ReluSquared's raw magnitude carries useful signal: mild regression ~0.001-0.003
- If sparsity-dependent RMS scaling causes issues: mild regression
- Interaction with c_proj zero-init: neutral (zero-init means MLP output is zero at step 0 regardless of norm)

## If it works

- Try adding norm after attention output too (before c_proj in attention) for full NormFormer
- Try norm with a learnable scale parameter (affine RMSNorm) for per-channel control
- Combine with z-loss 2e-4 (exp76 got 1.078430, might cross threshold with better MLP gradients)

## If it fails

- ReluSquared's magnitude information is useful, norm destroys it
- Try ReGLU at iso-compute (relu gating = sparsity + gating, never tested)
- Try Differential Transformer (attention subtraction between head pairs)
- Try increasing skip_lambdas init from 0.0 to 0.1 for faster U-Net skip learning
