Multi-Token Prediction (MTP) retry — auxiliary loss predicting token i+2 via learned projection + shared lm_head (exp10 crashed due to code bug).

## What changed
- Added `self.mtp_proj = nn.Linear(n_embd, n_embd, bias=False)` (512×512 = 262K params, 0.3% of total)
- In `forward()`, during training only (reduction='mean'), computes:
  - `mtp_hidden = mtp_proj(x[:, :-1])` — project hidden states at positions 0..T-2
  - `mtp_logits = lm_head(mtp_hidden)` — reuse existing output projection
  - Applies same softcap=15 to MTP logits
  - `mtp_loss = CE(mtp_logits, targets[:, 1:])` — predict token i+2 from position i
  - `loss = loss + 0.1 * mtp_loss` — conservative auxiliary weight
- mtp_proj initialized with same uniform distribution as other matrix params
- Added to Muon optimizer (grouped by shape with other 512×512 params)
- Evaluation path completely unchanged (MTP only runs when reduction='mean')

## Why
- **Exp10 crashed due to code bug, not a fundamental issue** — MTP was never actually tested
- **Meta MTP paper + DeepSeek-V3**: Predicting further ahead forces richer representations that capture longer-range dependencies. Each step extracts more learning signal from the same data.
- **Lightweight**: Only 262K extra params (0.3%). The compute overhead comes from the extra lm_head matmul (~10-15% wall-clock).
- **Shared lm_head**: No new embedding parameters. MTP gradient also improves lm_head quality.
- **Complementary to z-loss**: Z-loss (exp9, kept) improves gradient flow through softcap. MTP improves gradient signal richness. They target different aspects of training efficiency.

## Key differences from exp10 (crashed)
- Cleaner implementation: mtp_proj added directly as a module attribute, not nested
- Proper inclusion in optimizer: added to all_matrix_params list before Muon grouping
- Correct parameter counting: mtp included in num_scaling_params
- Careful tensor indexing: x[:, :-1] and targets[:, 1:] with .contiguous()

## Expected outcome
- If MTP enriches representations enough to offset ~10-15% fewer steps: improvement of ~0.001-0.005 in val_bpb
- If the compute overhead dominates: regression of ~0.002-0.005 (fewer steps, insufficient MTP signal)
- If effects cancel: neutral within noise (~0.003)
- No risk of OOM (only 262K extra params, ~1MB)


## Résultats
- val_bpb: 1.093492
- peak_vram_mb: 12569.6
- status: discarded
