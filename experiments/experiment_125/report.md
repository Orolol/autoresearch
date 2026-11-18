Increase UNEMBEDDING_LR from 0.004 to 0.01 (2.5x) to improve lm_head convergence — 22% of model params, never tuned.

## What changed

Single hyperparameter change in `train.py` line 501:
- `UNEMBEDDING_LR = 0.004` → `UNEMBEDDING_LR = 0.01`

Actual LR after dmodel scaling: 0.01 × (512/768)^-0.5 = 0.01 × 1.225 = 0.0123.

## Why

- **lm_head is 22% of model**: 512 × 32768 = 16.8M params, the single largest parameter matrix. With LR=0.004, it trains with a budget 150x below EMBEDDING_LR=0.6. In ~830 steps, this may severely under-train the output projection.
- **Dense gradients**: Unlike embeddings (sparse, only updated for tokens in the batch), lm_head gets a gradient from every token position in every step. This means AdamW moments converge quickly, and a higher LR is safe.
- **Never tested**: Piste #24 in prompt.md. EMBEDDING_LR was tuned early on (big gains), but UNEMBEDDING_LR was left at its default value.
- **Softcap safety net**: Even if the higher LR causes briefly larger logits, softcap=13 + z-loss=1e-4 constrain the logit range, preventing instability.

## Risks

- **Neutral (medium)**: Dense gradients may mean 0.004 was already well-scaled
- **Regression (low)**: Higher LR could over-update lm_head, especially for rare tokens
- **Crash (none)**: Just a scalar change, no structural modification
- Expected range: -0.005 to +0.003

## If it works

- Try UNEMBEDDING_LR=0.02 for stronger effect
- Try UNEMBEDDING_LR=0.05 to approach embedding LR territory
- Combine with next architectural idea (FP8 retry or DeltaNet)

## If it fails

- UNEMBEDDING_LR tuning is dead, 0.004 was already optimal
- Move to FP8 retry with torchao or simpler approach
- Consider DeltaNet/GLA as the only remaining bold architectural change


## Résultats
- val_bpb: 1.071063
- peak_vram_mb: 10445.0
- status: discarded
