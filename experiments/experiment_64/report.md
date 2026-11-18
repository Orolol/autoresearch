Add tiny LR warmup (WARMUP_RATIO=0.0 to 0.01) to stabilize the initial gradient phase after compilation.

## What changed

Single constant change: `WARMUP_RATIO = 0.0` -> `WARMUP_RATIO = 0.01` (line 462 of train.py).

This adds a linear LR warmup covering 1% of the 300s time budget (~3 seconds, ~8-9 optimizer steps).

## Why

Currently there is NO learning rate warmup. After the first 10 steps (compilation warmup, excluded from timing), the model jumps from random initialization straight to full learning rate. This is problematic because:

1. **Chaotic initial gradients.** A randomly initialized transformer produces high-variance, poorly-directed gradients. Full LR on these gradients causes large, noisy parameter updates that can push the model away from good basins.

2. **AdamW state initialization.** The exponential moving averages (exp_avg, exp_avg_sq) start at zero. The first few updates are essentially pure gradient descent (before the averages accumulate). With full LR, these unsmoothed updates can be destructively large for embedding/lm_head/scalar parameters.

3. **Standard practice.** Every major LLM training setup (GPT-2, GPT-3, LLaMA, PaLM) uses LR warmup. This codebase is the exception.

4. **Zero effective cost.** The warmup covers only ~8-9 steps at progressively increasing LR (from 0 to full). These steps still train the model — just with a gentler ramp. The 3s warmup is negligible vs the 300s budget.

5. **Marked HIGH PRIORITY** in prompt.md notes as the top untested change after z-loss tuning.

## Context from recent experiments

- Exp63 (z-loss 5e-5): 1.080565 (+0.002) — weaker z-loss hurt, confirming z-loss gradient is useful
- All Muon optimizer tweaks are confirmed DEAD (warmup speed, beta2, momentum ceiling)
- Gradient clipping DEAD (+0.010), HEAD_DIM=64 DEAD (+0.012), parallel attn+MLP DEAD (+0.034)
- This is the simplest remaining untested change with clear theoretical backing

## Risks

- Zero crash/OOM risk (no new parameters, no structural changes)
- Zero speed impact (same number of steps, just first ~8-9 at reduced LR)
- If warmup is unnecessary: identical to baseline within noise (warmup phase is tiny)
- If 0.01 is too short: neutral result suggests trying 0.02 or 0.03 next
- If warmup actively hurts: would mean the model benefits from "shock" initialization, very unlikely


## Résultats
- val_bpb: 1.082924
- peak_vram_mb: 10337.7
- status: discarded
