Increase Muon beta2 from 0.95 to 0.99 for smoother NorMuon per-row step sizing over a 100-step averaging window.

## What changed
- `beta2=0.95` -> `beta2=0.99` in `setup_optimizer` Muon param groups. Single number change.

## Why
- **NorMuon's second momentum buffer tracks per-row gradient variance** for adaptive step sizing. beta2 controls how fast this estimate adapts: beta2=0.95 gives a ~20-step effective window, beta2=0.99 gives ~100-step window.
- **Smoother per-row estimates reduce noise.** With ~830 total training steps, a 20-step window (beta2=0.95) is very responsive but noisy — per-row step sizes fluctuate rapidly. A 100-step window (beta2=0.99) provides more stable per-parameter learning rate adaptation.
- **Total step magnitude is NOT affected by beta2.** NorMuon normalizes the total gradient norm after per-row scaling (`final_scale = step_size * (v_norm / v_norm_new)`). So beta2 only changes HOW the gradient budget is distributed across matrix rows, not the total step size. This makes the change inherently safe.
- **Analogy to Adam beta2.** In Adam, beta2=0.999 (not 0.95) is standard because smoother denominator estimates lead to more stable training. The same principle applies to NorMuon's per-row variance tracking.
- **Never tested.** All 40 previous experiments used beta2=0.95. This is the first Muon second momentum experiment.

## No-bias-correction consideration
- NorMuon does NOT apply bias correction to second_momentum_buffer. At step 1, the buffer underestimates true variance by factor (1-beta2). With beta2=0.99, this means 100x underestimate → sqrt(100)=10x overestimate in step_size.
- **But this is neutralized by NorMuon's norm preservation.** The final_scale factor rescales everything so total gradient norm equals the original. The bias only affects the RELATIVE per-row distribution at initialization — it converges within ~100 steps (12% of training).
- With beta2=0.95, convergence happened within ~20 steps (2.4% of training). The slower convergence with 0.99 costs ~80 extra steps of suboptimal per-row distribution, but smoother estimates for the remaining 730 steps should compensate.

## Expected outcome
- If smoother per-row estimates help convergence: improvement of ~0.001-0.003
- If 20-step window was already fine: neutral within noise (~0.003)
- If slower adaptation hurts (gradient landscape changes faster than 100-step window): mild regression (~0.001-0.003)
- No crash, OOM, speed, or memory risk — pure scalar change in optimizer config
