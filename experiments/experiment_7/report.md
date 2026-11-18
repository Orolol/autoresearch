Increase MATRIX_LR from 0.04 to 0.06 (+50%) to accelerate convergence in the undertrained regime.

## What changed
- `MATRIX_LR = 0.04` → `MATRIX_LR = 0.06` (single line change)
- This increases the learning rate for all transformer matrix parameters (attention Q/K/V/proj + MLP fc/proj) optimized by Muon
- No change to embedding, unembedding, scalar, or VE learning rates

## Why
- The model is severely undertrained (~830 optimizer steps in 5 minutes)
- We confirmed that more optimizer steps help (2^18 batch size beat original baseline), but pushing further (2^17) failed due to gradient noise
- Alternative approach: instead of more steps, make each step more effective with a higher LR
- Muon's Newton-Schulz orthogonalization normalizes the gradient direction — the LR only controls step magnitude in an already-optimal direction
- Multiple stability mechanisms prevent divergence: QK-norm (prevents attention score explosion), softcap=15 (prevents logit explosion), cautious weight decay (prevents weight growth)
- 50% LR increase (0.04→0.06) is moderate: well within the stable region for Muon, which is known to tolerate higher LRs than Adam due to its gradient normalization

## Expected outcome
- If model was understepping: improvement of ~0.002-0.005 in val_bpb from faster convergence
- If 0.04 was already optimal: neutral result within noise (~0.003)
- If 0.06 overshoots: regression of ~0.003-0.010 from oscillation during warmdown phase
- No change in speed, memory, or parameters — pure training dynamics experiment
