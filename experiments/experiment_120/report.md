VE betas (0.9, 0.95) — increase VE Adam beta1 from 0.8 to 0.9 for smoother gradient EMA on sparsely-updated Value Embeddings.

## What changed

One-value change in `train.py` (GPT.setup_optimizer, line 292):

Before: `dict(kind='adamw', params=value_embeds_params, ..., betas=adam_betas, ...)`  (adam_betas = (0.8, 0.95))
After: `dict(kind='adamw', params=value_embeds_params, ..., betas=(0.9, 0.95), ...)`

No other changes. Architecture, init, LR, WD, all other optimizer groups unchanged.

## Why

- **VE receives uniquely sparse gradients**: Only 2 out of 8 layers (first+last) produce VE gradients. Each VE vector only gets updated when its token appears in-batch. With beta1=0.8, the gradient EMA decays 20% per step — after just 5 steps without seeing a token, its EMA is at 33% of original signal. With beta1=0.9, decay is 10% per step — after 5 steps, EMA is at 59%.
- **Complements VE WD=0.002 (exp115)**: WD regularizes VE norms (what), betas control gradient quality (how). These are orthogonal improvements.
- **Other AdamW groups don't need this**: wte/lm_head get dense gradients every step, so beta1=0.8 (fast adaptation) is appropriate. VE's sparse signal benefits from slower, smoother averaging.
- **Standard practice for sparse embeddings**: Higher beta1 for sparse parameters is established (e.g., BERT's embedding fine-tuning uses beta1=0.9 vs classifier beta1=0.8).

## Risks

- VE gradients may already be well-estimated at beta1=0.8 → neutral result (most likely)
- Higher beta1 means slower adaptation — if VE needs fast learning early, this could slow convergence slightly → mild regression
- Zero crash/OOM risk. Same code paths, just a different float value.
- Expected range: -0.002 to +0.002


## Résultats
- val_bpb: 1.069504
- peak_vram_mb: 10449.4
- status: discarded
