Reduce WARMDOWN_RATIO from 0.6 to 0.4 to spend 60% of training at full learning rate instead of 40%.

## What changed
- `WARMDOWN_RATIO = 0.6` -> `WARMDOWN_RATIO = 0.4` (single line change)
- This means the model trains at full LR for 60% of the time budget (was 40%), then linearly decays to 0 over the remaining 40%
- With ~830 total steps: ~500 steps at full LR (was ~330), ~330 steps in warmdown (was ~500)

## Why
- The model is severely undertrained in 5 minutes. Every step at full LR counts.
- Currently 60% of training is spent in warmdown — the model is "cooling down" for most of training. This seems overly conservative.
- More steps at full LR = more effective learning before the LR decays.
- The warmdown compression (40% instead of 60%) still provides ~330 steps to converge to a good minimum, which should be sufficient.
- MTP (exp12) failed due to compute overhead. Architectural changes have mostly regressed. This is a safe, orthogonal hyperparameter test.
- The linear warmdown from 1.0 to 0.0 over 40% of training gives a steeper but still smooth decay curve.

## Expected outcome
- If the model benefits from more full-LR steps: improvement of ~0.001-0.003 in val_bpb
- If the current warmdown ratio was already optimal: neutral within noise (~0.003)
- If the compressed warmdown doesn't give enough time to settle: regression of ~0.001-0.003
- No crash, OOM, or instability risk — pure schedule change


## Résultats
- val_bpb: 1.089407
- peak_vram_mb: 11412.5
- status: discarded
