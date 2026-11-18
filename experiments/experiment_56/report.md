Increase Muon beta2 from 0.95 to 0.99 for smoother NorMuon per-row adaptive step sizing.

## What changed
- `beta2=0.95` -> `beta2=0.99` in `setup_optimizer` Muon param groups (line 273). Single number change.

## Why
- **NorMuon's second momentum buffer tracks per-row gradient variance.** With beta2=0.95, the effective averaging window is ~20 steps. With beta2=0.99, it's ~100 steps. Smoother estimates reduce noise in per-row adaptive step sizes, leading to more consistent optimization.
- **Analogous to standard Adam practice.** Adam's default beta2 is 0.999, not 0.95. The 20-step window in the current Muon config may cause noisy per-row scaling, especially early in training when gradient statistics are unstable.
- **NorMuon's norm preservation makes this inherently safe.** Beta2 only affects per-row budget distribution within a fixed total gradient norm. The total step magnitude is invariant to beta2 — only the distribution of step size across rows changes. No risk of training instability.
- **100-step window is still responsive.** With ~830 training steps, a 100-step effective window can adapt to changing gradient statistics within ~12% of training. Not too sluggish.
- **Never tested on GPU.** Attempted 3 times (exp41/42/50) but ALL crashed on RunPod infra (not code issues). This is the 4th attempt and the #1 recommended untested change per prompt.md notes.

## Expected outcome
- If smoother per-row estimates help: improvement of ~0.0005-0.002 (val_bpb ~1.077-1.078)
- If 20-step window was fine: neutral within noise (~0.001)
- If fast adaptation was important: mild regression ~0.001-0.003
- No crash, OOM, speed, or memory risk — pure scalar change in optimizer config
