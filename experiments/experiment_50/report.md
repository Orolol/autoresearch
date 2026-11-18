Increase Muon beta2 from 0.95 to 0.99 for smoother NorMuon per-row adaptive step sizing over a ~100-step averaging window.

## What changed
- `beta2=0.95` -> `beta2=0.99` in `setup_optimizer` Muon param groups. Single number change.

## Why
- **NorMuon's second momentum buffer tracks per-row gradient variance.** With beta2=0.95, the effective averaging window is ~20 steps. With beta2=0.99, it's ~100 steps. Smoother estimates reduce noise in per-parameter adaptive step sizes.
- **Analogous to standard Adam practice.** Adam uses beta2=0.999 (not 0.95) as default. The shorter window in Muon may cause noisy per-row scaling, especially early in training when gradient statistics are unstable.
- **NorMuon's norm preservation makes this inherently safe.** Beta2 only affects per-row budget distribution within a fixed total gradient norm. The total step magnitude is invariant to beta2 — only the distribution of step size across rows changes. This means there's no risk of training instability from this change.
- **100-step window is still responsive.** With ~830 training steps, a 100-step effective window can adapt to changing gradient statistics within ~12% of training. Not too sluggish.
- **Never tested on GPU.** Attempted twice (exp41/42) but both crashed on RunPod infra (not code issues). Infra is now stable (exp48 ran successfully with softcap=11).

## Expected outcome
- If smoother per-row estimates help: improvement of ~0.001-0.003
- If 20-step window was fine: neutral within noise (~0.003)
- If fast adaptation was important: mild regression ~0.001-0.003
- No crash, OOM, speed, or memory risk — pure scalar change in optimizer config
