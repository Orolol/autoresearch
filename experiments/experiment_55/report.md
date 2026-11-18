Reduce Muon momentum warmup from 300 to 100 steps to reach full 0.95 Nesterov momentum faster.

## What changed
- `get_muon_momentum`: `min(step / 300, 1)` → `min(step / 100, 1)`. Single number change.

## Why
- **36% → 12% of training at suboptimal momentum.** With ~830 training steps, 300-step warmup means 36% of training runs at momentum < 0.95. Reducing to 100 steps means only 12%.
- **2.5x stronger Nesterov look-ahead at step 100.** At step 100, new schedule gives momentum=0.950 vs old=0.883. Look-ahead distance = momentum/(1-momentum): 19.0 vs 7.6. The optimizer "sees further ahead" during the critical middle phase of training.
- **NS + NorMuon provide stability.** Newton-Schulz orthogonalization normalizes gradient direction independently of momentum. NorMuon normalizes per-row step size via second moment tracking. Higher momentum only changes direction weighting — step magnitude is controlled separately.
- **Never tested on GPU.** Attempted 4+ times (exp38/40/49/51) but all crashed on RunPod infra. The code was verified correct each time. This is the most impactful untested change.

## Expected outcome
- If faster warmup helps: improvement of ~0.0005-0.001 (val_bpb ~1.077-1.078)
- If 300-step warmup was optimal: neutral within noise (~0.001)
- If too fast (early instability): mild regression ~0.001-0.003
- No crash, OOM, speed, or memory risk — pure scalar change in a schedule function


## Résultats
- val_bpb: 1.079176
- peak_vram_mb: 10337.7
- status: discarded
