Increase Muon Nesterov momentum ceiling from 0.95 to 0.97 for 70% stronger gradient look-ahead (retry of exp52 which crashed on infra).

## What changed

Single number change in `get_muon_momentum`: the momentum ceiling goes from 0.95 to 0.97.

```python
# Before
return (1 - frac) * 0.85 + frac * 0.95

# After
return (1 - frac) * 0.85 + frac * 0.97
```

## Why

Nesterov look-ahead distance = momentum / (1 - momentum):
- At 0.95: distance = 19.0
- At 0.97: distance = 32.3 (+70%)

The optimizer "sees further ahead" in the loss landscape, potentially finding better descent directions. This is a different axis than momentum warmup speed (exp55 tested 300->100 steps) — it changes the steady-state momentum after warmup completes.

Safety: NS orthogonalization + NorMuon normalize step magnitude independently. Momentum only affects the direction weighting of the Nesterov extrapolation. Standard Nesterov range is 0.9-0.99; 0.97 is conservative.

Exp52 attempted this exact change but crashed on infra (never tested on GPU). Marked HIGH PRIORITY in prompt.md notes.

## Risks

- Zero crash/OOM/speed risk (single scalar change)
- If 0.97 overshoots: mild regression ~0.001-0.003
- More inertia during warmdown phase could either help (smoother convergence) or hurt (slower adaptation to decaying LR)


## Résultats
- val_bpb: 1.082534
- peak_vram_mb: 10337.7
- status: discarded
