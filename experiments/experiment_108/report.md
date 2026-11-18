VE embedding LR separation — lower VE LR from 0.6 to 0.2 for more stable value reference signal.

## What changed

Three minimal changes in `train.py`:

1. **New hyperparameter** (line 497): `VE_LR = 0.2` — dedicated learning rate for value embeddings, 3x lower than `EMBEDDING_LR = 0.6`.

2. **setup_optimizer signature** (line 275): Added `ve_lr=0.2` parameter. The VE param group now uses `ve_lr * dmodel_lr_scale` instead of `embedding_lr * dmodel_lr_scale`.

3. **Optimizer call** (line 569): Passes `ve_lr=VE_LR` to `setup_optimizer`.

## Why

- **VE bookend was the biggest win** (-0.004, exp84). VE is the most impactful architectural component — optimizing its training dynamics has high expected value.
- **Different roles, different LRs**: Token embeddings (wte) are the primary input representation — every token must learn a good representation quickly in ~830 steps. VE serves a fundamentally different purpose: providing a stable token identity reference to value computation in the first and last layers. A slower-moving VE is a better reference signal.
- **VE gates already modulate contribution**: The learned VE gates (sigmoid-based, starting neutral at 1.0) control how much VE contributes to V. If VE itself is changing rapidly alongside the gates, the gate has a moving target. A more stable VE embedding lets the gates learn their optimal modulation more cleanly.
- **Last 3 experiments all failed with architectural changes**: exp105 (DiffAttn, +0.009), exp106 (sandwich norm, +0.008), exp107 (depthwise conv, +0.023). Time to optimize the existing winning architecture rather than adding new components.

## Risks

- LR=0.2 might be too low — VE might not adapt fast enough, providing weak signal
- LR=0.2 might still be too high — optimal could be 0.1 or lower
- Current LR=0.6 might already be near-optimal for VE
- Expected range: -0.003 to +0.004

## If it works

- Try VE_LR=0.1 or 0.05 for even more stable reference
- Try VE weight decay (currently 0.0) for additional regularization
- Try different VE betas (currently same as other AdamW: 0.8, 0.95)
- Combine with wider multi-scale RoPE or Schedule-Free Adam

## If it fails

- VE LR=0.6 is already good. Mark VE LR separation as dead.
- Try wider multi-scale RoPE range (option 4)
- Try Schedule-Free Adam for AdamW groups (option 2)
- Try FP8 matmuls on Blackwell (option 3)


## Résultats
- val_bpb: 1.070530
- peak_vram_mb: 10450.8
- status: discarded
