Initialize skip_lambdas to 0.05 instead of 0.0 for immediate U-Net skip connection activation.

## What changed

Single value change in `GPT.init_weights()` (line 188): `self.skip_lambdas.fill_(0.05)` instead of `self.skip_lambdas.fill_(0.0)`. This makes U-Net skip connections immediately active from step 0 instead of requiring the optimizer to learn non-zero values from scratch.

## Why

U-Net skip connections (exp71) were the second biggest win: -0.001 val_bpb, connecting early layer outputs (0,1,2,3) to mirror late layers (7,6,5,4) with learnable scalars. But these scalars start at 0.0, meaning:

1. For the first N steps, skip connections contribute nothing (as if they don't exist)
2. The Adam optimizer with beta1=0.96 and scalar_lr=0.5 is slow to move scalars away from zero — the momentum buffer needs time to build up
3. With only ~830 training steps total, every step without active skips is wasted potential

Setting skip_lambdas to 0.05 means:
- Immediate gradient flow shortcuts from early to late layers
- Late layers receive early-layer features from step 0
- The optimizer can fine-tune the values up or down rather than discovering them from zero
- 0.05 is conservative: the skip contribution (0.05 * skip_state) is small relative to resid_lambdas (1.0 * x) and x0_lambdas (0.1 * x0)

## Risks

- Zero crash/OOM risk (single scalar init change, no new code)
- Zero compute cost (no new operations)
- If early-layer representations are low quality at init (they are — random weights), the 0.05 skip adds noise. But 0.05 is very small and the optimizer can reduce it if harmful.
- If the optimal skip_lambdas values are very different across layers, the uniform 0.05 init might not help as much as layer-specific values.

## If it works

- Try larger init (0.1, 0.15) for even faster skip activation
- Try asymmetric init: higher for the deepest skip (layer 0→7) and lower for shallow skips (layer 3→4)
- Combine with z-loss 2e-4 (directional improvement confirmed in exp76: 1.078430)

## If it fails

- 0.05 was too high — try 0.02 (more conservative)
- Skip connections benefit from learning from zero (the optimizer finds the right scale better when starting from zero)
- Try Differential Transformer (attention subtraction between head pairs)
- Try adding VE to first + last layer (more value embedding coverage)
