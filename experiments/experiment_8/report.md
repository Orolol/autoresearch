Re-run MATRIX_LR=0.06 (+50%) which crashed in exp7 due to RunPod API timeout (never actually tested).

## What changed
- `MATRIX_LR = 0.04` -> `MATRIX_LR = 0.06` (single line change)
- This increases the learning rate for all transformer matrix parameters (attention Q/K/V/proj + MLP fc/proj) optimized by Muon
- No change to embedding, unembedding, scalar, or VE learning rates

## Why
- Exp7 attempted this exact change but crashed due to RunPod API timeout — the code was never executed on GPU
- The model is severely undertrained (~830 optimizer steps in 5 minutes)
- We can't get more steps: 2^17 batch size failed (gradient noise, exp6: +0.011 regression)
- Alternative approach: make each step more effective with a higher LR
- Muon's Newton-Schulz orthogonalization normalizes gradient direction — higher LR just means bigger steps in an already-optimal direction
- Multiple stability mechanisms prevent divergence: QK-norm, softcap=15, cautious weight decay
- 50% LR increase is moderate for Muon, which tolerates higher LRs than Adam

## Expected outcome
- If model was understepping: improvement of ~0.002-0.005 in val_bpb
- If 0.04 was already optimal: neutral result within noise (~0.003)
- If 0.06 overshoots: regression of ~0.003-0.010 from oscillation during warmdown
- No change in speed, memory, or parameters


## Résultats
- val_bpb: 1.088767
- peak_vram_mb: 11412.5
- status: discarded
