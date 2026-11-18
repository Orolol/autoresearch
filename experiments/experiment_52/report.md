Increase Muon Nesterov momentum ceiling from 0.95 to 0.97 for stronger gradient look-ahead during training.

## What changed
- `get_muon_momentum`: `0.95` -> `0.97` (momentum ceiling). Single number change.

## Why
- **Stronger Nesterov look-ahead.** Momentum 0.97 gives look-ahead distance momentum/(1-momentum) = 32.3, vs 19.0 at 0.95. That's 70% stronger gradient anticipation, helping the optimizer navigate the loss landscape more efficiently.
- **NS + NorMuon provide stability.** Newton-Schulz orthogonalization normalizes gradient direction independently of momentum. NorMuon normalizes per-row step size via second moment tracking. Higher momentum only changes the direction weighting between current and past gradients — the step magnitude is controlled separately. This makes the change inherently safe.
- **More inertia during warmdown.** During the 60% warmdown phase, higher momentum helps the optimizer maintain useful trajectory information as the learning rate decreases. At 0.97, the effective averaging window is ~33 steps vs ~20 steps at 0.95.
- **Never tested before.** All previous Muon momentum experiments targeted the warmup schedule (300 vs 100 steps), not the ceiling value. The ceiling has been 0.95 since experiment 0.

## Expected outcome
- If 0.97 improves look-ahead: improvement of ~0.001-0.002 (val_bpb ~1.077-1.078)
- If 0.95 was already optimal: neutral within noise (~0.001)
- If 0.97 is too aggressive (overshoot past good minima): mild regression ~0.001-0.003
- No crash, OOM, speed, or memory risk — pure scalar change in a schedule function
