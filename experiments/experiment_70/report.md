Decouple AdamW/Muon LR schedules — remove warmdown for AdamW groups while keeping Muon warmdown intact.

## What changed

In the training loop's schedule section, the LR warmdown multiplier (`lrm`) is now applied only to Muon param groups. AdamW groups (embeddings, lm_head, value_embeds, resid_lambdas, x0_lambdas) keep their initial LR constant throughout training.

3-line change in the schedule loop (lines 586-591).

## Why

The current warmdown (WARMDOWN_RATIO=0.6, FINAL_LR_FRAC=0.0) decays ALL groups' LR to zero over the last 60% of training. This means:
- Only ~330 out of ~830 steps run at full LR for embeddings and lm_head
- The embedding tables (wte: 32768x512, VE: 32768x512 = 33.6M params, 44% of total) are severely undertrained
- These tables are updated sparsely (each token's embedding only updates when that token appears in the batch)

AdamW's adaptive step sizes already self-regulate: the second moment (exp_avg_sq) tracks per-parameter gradient variance, naturally reducing effective step sizes as gradients stabilize. External warmdown is redundant for AdamW.

Muon (matrix params) genuinely needs warmdown because NS orthogonalization produces constant-magnitude updates — without LR decay, it would keep making large steps.

Exp13 tested WARMDOWN_RATIO=0.4 for ALL groups and regressed (1.089407). But that included Muon, which needs warmdown. This experiment specifically decouples the schedules.

## Risks

- Zero crash/OOM risk (schedule logic change, no structural modification)
- If embeddings benefit from LR decay (less oscillation near end): mild regression ~0.001-0.003
- If Muon-AdamW schedule mismatch causes representational drift: mild regression
- If constant LR helps embeddings learn better: improvement ~0.0005-0.002

## If it works

- Try also removing warmdown for just the embedding groups (LR=0.6) but keeping it for lm_head (LR=0.004) — test which group benefits most
- Try a partial warmdown for AdamW: FINAL_LR_FRAC=0.5 (decay to 50% instead of 0)
- Combine with z-loss 2e-4 (stronger regularization to compensate for constant LR)

## If it fails

- The warmdown is important for ALL groups. Schedule decoupling is dead.
- Try z-loss coefficient 2e-4 (increase from 1e-4, follows the positive signal from exp9 vs exp63)
- Try Schedule-Free Adam (full implementation with Polyak averaging, not just constant LR)
- Move to more radical architectural changes: simple 2-expert MoE, differential attention


## Résultats
- val_bpb: 1.090516
- peak_vram_mb: 10337.7
- status: discarded
