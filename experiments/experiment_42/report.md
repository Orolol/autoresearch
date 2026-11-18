Retry Muon beta2=0.99 (exp41 crashed infra, never tested on GPU) for smoother NorMuon per-row step sizing.

## What changed
- `beta2=0.95` -> `beta2=0.99` in `setup_optimizer` Muon param groups. Single number change.

## Why
- **NorMuon's second momentum buffer tracks per-row gradient variance** for adaptive step sizing. beta2=0.95 gives a ~20-step effective window, beta2=0.99 gives ~100-step window. Smoother estimates = more stable per-parameter learning rate adaptation.
- **Total step magnitude is NOT affected by beta2.** NorMuon normalizes the total gradient norm after per-row scaling. So beta2 only changes HOW the gradient budget is distributed across matrix rows, not the total step size. This makes the change inherently safe.
- **Analogy to Adam beta2.** In Adam, beta2=0.999 (not 0.95) is standard because smoother denominator estimates lead to more stable training. Same principle applies.
- **Exp41 crashed on RunPod infra** — code was correct (identical single-number change). This is a pure retry.
- **Never tested on GPU.** All 41 previous experiments used beta2=0.95. First successful Muon second momentum test.

## Expected outcome
- If smoother per-row estimates help convergence: improvement of ~0.001-0.003
- If 20-step window was already fine: neutral within noise (~0.003)
- If slower adaptation hurts: mild regression (~0.001-0.003)
- No crash, OOM, speed, or memory risk — pure scalar change in optimizer config
