VE weight decay 0.002 — add mild L2 regularization to Value Embedding parameters to prevent rare token VE drift.

## What changed

One-value change in `train.py` (GPT.setup_optimizer, line 292):

Before: `dict(kind='adamw', params=value_embeds_params, ..., weight_decay=0.0)`
After: `dict(kind='adamw', params=value_embeds_params, ..., weight_decay=0.002)`

No other changes. Architecture, init, LR, all other optimizer groups unchanged.

## Why

- **VE is the largest unregularized parameter block**: 16.8M params (22% of model) with weight_decay=0.0. Meanwhile, transformer matrices use WD=0.2 (cautious Muon). VE is the only large parameter block with zero regularization.
- **Rare token VE drift**: With 32K vocab and ~830 steps at batch size 262K, rare tokens appear few times. Their VE vectors receive sparse gradient updates at high LR (0.6), which can cause norm growth. WD pushes unused vectors toward zero — a better default than uncontrolled drift.
- **VE is auxiliary, not primary**: The main value signal comes from c_v (contextual). VE adds token identity as a bonus. Mild regularization encourages the model to rely on the contextual signal for rare tokens rather than a potentially noisy VE vector.
- **Effective decay**: With LR=0.6 and WD=0.002, per-step factor = 1 - 0.0012 = 0.9988. Over the full-LR phase (~332 steps): 0.9988^332 ~ 0.67. During warmdown, decay weakens as LR drops. Total VE norm reduction ~30-35% — meaningful but not aggressive.

## Risks

- WD=0.002 might be too mild to matter → neutral (most likely outcome)
- WD might regularize away useful VE signal for rare but important tokens → mild regression
- Zero crash/OOM risk. Exact same code paths, just a different float value.
- Expected range: -0.002 to +0.002

## If it works

- Try WD=0.005 for stronger regularization
- Combine with VE betas tuning (beta1=0.9 for smoother VE updates)
- Try adding WD to wte embedding too (currently 0.0)

## If it fails

- VE doesn't need regularization at this training length. Mark VE WD as dead.
- Try VE betas (0.9, 0.95) for smoother VE updates
- Try VE_LR=0.4 (midpoint between exp108's 0.2 and current 0.6)
- Try FP8 matmuls on Blackwell for 2x throughput (the "more steps" lever)


## Résultats
- val_bpb: 1.069319
- peak_vram_mb: 10449.4
- status: IMPROVED
