Initialize skip_lambdas to 0.05 instead of 0.0 for immediate U-Net skip connection activation from step 1.

## What changed

One-line change in `init_weights()`: `self.skip_lambdas.fill_(0.0)` changed to `self.skip_lambdas.fill_(0.05)`.

This means U-Net skip connections (which connect early layer outputs to mirror late layers: 0->7, 1->6, 2->5, 3->4) are immediately active at 5% strength from the first training step, instead of starting completely inactive and relying on the optimizer to learn to activate them.

## Why

U-Net skip connections were the 3rd most impactful architectural win (exp71, -0.001 val_bpb). Currently initialized to 0.0, the skip connections are completely inactive initially. With only ~830 training steps in the 300s budget, parameters initialized closer to their useful operating range converge faster. At 0.05, each late mirror layer immediately receives a 5% contribution from its early mirror, providing gradient shortcuts from step 1.

This idea was listed as priority #6 in remaining untested ideas. Exp82 attempted this but crashed on RunPod infra — it has never been tested on GPU.

## Risks

- Very low risk. 0.05 is a small perturbation; the optimizer (AdamW with lr=0.5 for scalars) can easily correct if wrong.
- No OOM risk, no extra compute, no torch.compile issues.
- Worst case: negligible regression within run-to-run noise (~0.001).

## Context

- DEPTH=9 confirmed worse in exp97 (1.079870 vs 1.072821 best) — step reduction outweighs capacity gain.
- Most architectural changes exhausted after 97 experiments.
- Remaining high-priority ideas: DeltaNet/GLA (complex), FP8 matmuls (complex), wider RoPE range (low expected impact).


## Résultats
- val_bpb: 1.070366
- peak_vram_mb: 10450.8
- status: IMPROVED
