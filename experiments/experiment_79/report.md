Mid-MLP normalization (NormFormer-style) — add RMSNorm between ReluSquared activation and output projection in MLP (3rd attempt, exp77/78 crashed on infra).

## What changed

Single line added in `MLP.forward()`: `x = norm(x)` after `x = F.relu(x).square()` and before `x = self.c_proj(x)`. This normalizes the sparse hidden activations via RMSNorm before they enter the output projection.

## Why

Every winning experiment in this series improved gradient flow:
- Softcap 13: better gradient through tanh (exp45, -0.001)
- Z-loss: logit calibration gradient (exp9, marginal)
- U-Net skips: shorter gradient paths (exp71, -0.001, current best)

The MLP path is the last major unnormalized internal activation. ReluSquared produces sparse activations (~50% zeros) with potentially large squared values. Without normalization, c_proj must handle varying input magnitudes as training progresses, complicating its optimization. Adding RMSNorm ensures c_proj always receives unit-variance inputs, simplifying its learning task and stabilizing gradient flow.

This is the NormFormer approach (Shleifer et al., 2021) — adding norms within sublayers, not just before them. Zero extra parameters, negligible compute cost (one RMSNorm per layer vs much larger matmuls).

Exp77 and exp78 attempted this exact change but both crashed on infra (0.0 peak_vram = never reached GPU). This is the 3rd attempt.

## Risks

- Zero crash/OOM risk (norm() is used throughout the model, well-supported by torch.compile)
- Zero speed impact (one cheap elementwise op per layer)
- If ReluSquared's raw magnitude carries useful signal: mild regression ~0.001-0.003
- If sparsity-dependent RMS scaling causes issues: mild regression
- Interaction with c_proj zero-init: neutral (MLP output is zero at step 0 regardless of norm)

## If it works

- Try adding norm after attention output too (before c_proj in attention) for full NormFormer
- Combine with z-loss 2e-4 (exp76 got 1.078430, might cross threshold with better MLP gradients)
- Try learnable affine scale in the mid-MLP norm for per-channel control

## If it fails

- ReluSquared's magnitude information is useful, norm destroys it
- Try ReGLU at iso-compute (relu gating, never tested)
- Try increasing skip_lambdas init from 0.0 to 0.1 for faster U-Net skip learning
- Try Differential Transformer (attention subtraction between head pairs)


## Résultats
- val_bpb: 1.079762
- peak_vram_mb: 10339.7
- status: discarded
