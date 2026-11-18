Z-loss regularization (PaLM-style) to improve logit calibration and gradient flow through the softcap layer.

## What changed
- Added z-loss term in the `forward()` method, only during training (when `reduction='mean'`)
- After computing cross-entropy loss, computes `logz = torch.logsumexp(logits, dim=-1)` and adds `1e-4 * logz.square().mean()` to the loss
- 3 lines of code added, no other changes
- Evaluation path (reduction='none') is completely unchanged

## Why
- **PaLM and ST-MoE proven technique**: Z-loss penalizes the log-partition function magnitude, encouraging well-calibrated logit distributions
- **Complementary to softcap**: The model uses `softcap=15 * tanh(logits/15)` to bound logits. But tanh saturates near ±15, causing vanishing gradients for logits near the boundary. Z-loss provides a smooth gradient signal pushing logits away from saturation, keeping them in the regime where tanh gradients flow well
- **Zero cost**: One `logsumexp` per batch is negligible compute. No memory, speed, or architecture changes
- **Well-calibrated coefficient**: With logits bounded to [-15, 15] by softcap and vocab_size=32768, the z-loss term contributes ~0.06 to the total loss (~2% of CE loss of ~3.0), which is a mild but meaningful regularization signal

## Expected outcome
- If z-loss helps gradient flow through softcap: improvement of ~0.001-0.005 in val_bpb
- If softcap is already sufficient: neutral result within noise (~0.003)
- Very unlikely to cause any regression — the coefficient is conservative (1e-4)
- No risk of crash, OOM, or instability


## Résultats
- val_bpb: 1.087971
- peak_vram_mb: 11412.5
- status: IMPROVED
