Multi-Token Prediction (MTP) — auxiliary loss predicting token i+2 from position i via learned projection + shared lm_head.

## What changed
- Added `self.mtp_proj = nn.Linear(n_embd, n_embd, bias=False)` (512x512 = 262K params, 0.4% of total)
- In `forward()`, during training only (reduction='mean'), computes:
  - `mtp_logits = lm_head(mtp_proj(x[:, :-1]))` — transform hidden states then predict 2-ahead token
  - Applies same softcap=15 to MTP logits
  - `mtp_loss = CE(mtp_logits, targets[:, 1:])` — targets shifted by 1 = token i+2 from position i
  - `loss = loss + 0.1 * mtp_loss` — conservative auxiliary weight
- mtp_proj initialized with same uniform distribution as other matrix params
- Added to Muon optimizer (same shape group as attention projections)
- Evaluation path completely unchanged (MTP only runs when reduction='mean')

## Why
- **Meta MTP paper + DeepSeek-V3**: Predicting further ahead forces the model to build richer representations that capture longer-range dependencies. Each training step extracts more learning signal from the same data.
- **Lightweight**: The projection is only 262K params. The compute overhead comes from the extra lm_head matmul (~10-15% wall-clock).
- **Shared lm_head**: No new embedding parameters. The auxiliary prediction reuses the existing unembedding matrix, so the MTP gradient also improves lm_head quality.
- **Complementary to z-loss**: Z-loss (exp9) improves gradient flow through softcap. MTP improves gradient signal richness. They target different aspects of training efficiency.
- **No eval impact**: MTP is purely a training regularizer. The mtp_proj is unused at eval time.

## Expected outcome
- If MTP enriches representations enough to offset ~10-15% fewer steps: improvement of ~0.001-0.005 in val_bpb
- If the compute overhead dominates: regression of ~0.002-0.005 (fewer steps, insufficient MTP signal)
- If effects cancel: neutral within noise (~0.003)
- No risk of crash, OOM, or instability
