Increase logit softcap from 15 to 30 to allow more expressive predictions while keeping training stable.

## What changed
- `softcap = 15` → `softcap = 30` in the forward method (line 294)
- No other changes to architecture, hyperparameters, or optimizer

## Why
- Previous experiments (exp11 from earlier series) showed that removing softcap entirely (softcap=∞) worsened val_bpb to 1.097215 (+0.009 vs baseline), indicating softcap provides useful regularization
- However, softcap=15 may be overly restrictive — it aggressively clips logits, potentially limiting the model's ability to make confident predictions during later training when the model has learned meaningful patterns
- softcap=30 is the natural middle ground: still prevents logit explosion but gives the model 2x more dynamic range for confident predictions
- This is especially relevant for the 5-minute training regime where the model needs to quickly develop sharp output distributions

## Expected outcome
- If softcap=15 was too restrictive: improvement of ~0.002-0.005 in val_bpb
- If softcap=15 was already well-tuned: neutral result (within noise ~0.003)
- Very unlikely to be worse unless training becomes unstable (softcap=30 still provides strong regularization)
- No change in speed, memory, or parameters — pure quality experiment


## Résultats
- val_bpb: 1.093279
- peak_vram_mb: 11412.5
- status: discarded
