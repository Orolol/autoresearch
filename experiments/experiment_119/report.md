VE weight decay 0.005 — increase VE L2 regularization from 0.002 to 0.005 to test if stronger rare-token drift prevention improves val_bpb.

## Change

Single value change in `setup_optimizer()`: VE param group `weight_decay=0.002` -> `weight_decay=0.005`.

## Rationale

Exp115 (VE WD=0.002) produced the current best at 1.069319 (-0.00088 vs previous best). This experiment probes whether the WD response curve has peaked at 0.002 or if stronger regularization helps further. The VE embedding (16.8M params, 22% of model) is shared between first and last layers and receives sparse gradient updates (only 2 layers, only tokens in-batch). Rare tokens (~6% of vocab per batch) accumulate drift in their VE vectors. Stronger WD pushes these harder toward zero (a better default than random drift) while common tokens resist the decay via strong gradients. Total decay over full-LR phase: ~0.43 (vs ~0.67 at WD=0.002). Still mild compared to matrix WD=0.2.

## Expected outcome

If 0.005 improves: VE WD optimum is above 0.002, try 0.01 next.
If 0.005 is neutral/worse: optimum is near 0.002, VE WD tuning is done.
Expected range: -0.002 to +0.003.


## Résultats
- val_bpb: 1.069415
- peak_vram_mb: 10449.4
- status: discarded
