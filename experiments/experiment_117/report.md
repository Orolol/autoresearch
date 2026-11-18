wte embedding weight decay 0.002 — add mild L2 regularization to token embeddings, mirroring VE WD success (exp115).

## What changed

One-value change in `train.py` (GPT.setup_optimizer, line 291):

Before: `dict(kind='adamw', params=embedding_params, ..., weight_decay=0.0)`
After: `dict(kind='adamw', params=embedding_params, ..., weight_decay=0.002)`

No other changes. Architecture, init, LR, all other optimizer groups unchanged.

## Why

- **Direct analog of exp115 (new best, -0.00088)**: VE WD=0.002 worked by regularizing rare token VE drift. wte has the exact same characteristics: 16.8M params, LR=0.6, WD=0.0, 32K vocab with sparse rare-token updates.
- **wte is the only remaining large unregularized parameter block**: VE now has WD=0.002 (exp115), transformer matrices have WD=0.2 (Muon). wte (22% of model) still has WD=0.0.
- **Effective decay matches VE**: LR=0.6 x WD=0.002 = 0.0012 per-step. Over ~332 full-LR steps: 0.9988^332 ~ 0.67 total norm reduction. Meaningful but not aggressive.
- **Explicitly listed as piste #17** in prompt.md: "wte embedding weight decay — same logic as VE WD (exp115). Untested."

## Risks

- wte gets gradient from every token via cross-entropy loss (unlike VE which only gets signal through VE layers). This means wte embeddings may be better calibrated and not need WD → neutral result.
- WD could regularize away useful rare-token representations → mild regression.
- Zero crash/OOM risk. Same code paths, just a different float value.
- Expected range: -0.002 to +0.002

## If it works

- Try WD=0.005 for stronger wte regularization
- Try matching WD=0.005 for both wte and VE
- Combine with VE betas tuning (beta1=0.9) for smoother embedding updates

## If it fails

- wte doesn't need WD (cross-entropy provides sufficient gradient signal). Mark wte WD as dead.
- Try VE WD=0.005 (stronger VE regularization, piste #12)
- Try VE betas (0.9, 0.95) for smoother VE updates (piste #13)
- Try FP8 matmuls on Blackwell for 2x throughput (piste #3)


## Résultats
- val_bpb: 1.070293
- peak_vram_mb: 10449.4
- status: discarded
