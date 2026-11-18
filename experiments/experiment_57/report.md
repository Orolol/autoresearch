Increase Muon beta2 from 0.95 to 0.99 for smoother NorMuon per-row adaptive step sizing.

## What changed
- `beta2=0.95` -> `beta2=0.99` in `setup_optimizer` Muon param groups (line 273). Single number change.

## Why
- **Smoother per-row gradient variance estimates.** NorMuon's second momentum buffer tracks per-row gradient variance to distribute step size across rows. With beta2=0.95, the effective averaging window is ~20 steps — noisy, especially early in training. With beta2=0.99, the window expands to ~100 steps, producing more stable per-row step sizes.
- **Analogous to standard Adam practice.** Adam defaults to beta2=0.999, not 0.95. The current 20-step window may cause noisy per-row scaling that hurts convergence quality.
- **NorMuon's norm preservation makes this inherently safe.** Beta2 only affects per-row budget distribution within a fixed total gradient norm. The total step magnitude is invariant to beta2. No risk of training instability or divergence.
- **100-step window is still responsive.** With ~830 training steps, a 100-step effective window can adapt to changing gradient statistics within ~12% of training. Not sluggish.
- **Never tested on GPU.** Attempted 4 times (exp41/42/50/56) but ALL crashed on RunPod infra (not code issues). This is the 5th attempt and the #1 recommended untested change per prompt.md notes.

## Expected outcome
- If smoother per-row estimates help: improvement of ~0.0005-0.002 (val_bpb ~1.077-1.078)
- If 20-step window was fine: neutral within noise (~0.001)
- If fast adaptation was important: mild regression ~0.001-0.003
- No crash, OOM, speed, or memory risk — pure scalar change in optimizer config


## Résultats
- val_bpb: 1.079416
- peak_vram_mb: 10337.7
- status: discarded
