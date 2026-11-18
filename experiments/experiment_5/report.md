Remove all weight decay (WEIGHT_DECAY=0.0) to test if Muon's cautious WD is counterproductive during short 5-minute training.

## What changed
- `WEIGHT_DECAY = 0.2` -> `WEIGHT_DECAY = 0.0` (single line change)
- This removes all weight decay from Muon optimizer groups
- The cautious WD masking (`mask = (g * stacked_params) >= 0`) still computes but multiplies by 0, so it's a no-op
- `get_weight_decay(progress)` returns 0 for all progress values
- No other changes to architecture or hyperparameters

## Why
- Weight decay is a regularization technique designed to prevent overfitting during long training
- With only 5 minutes / ~830 optimizer steps, overfitting is unlikely — the model is severely undertrained
- The current WD implementation is already very mild: cautious masking (only penalizes aligned weight-gradient pairs) AND linear decay to 0 over training
- But even this mild WD slows down the effective learning rate for matrix parameters during the critical first 40% of training (before warmdown)
- Removing WD entirely lets the optimizer take larger effective steps, potentially improving convergence in the limited time budget
- This is explicitly listed as untested in the notes

## Expected outcome
- If WD was hurting convergence: improvement of ~0.002-0.005 in val_bpb
- If WD was already negligible: neutral result within noise (~0.003)
- Very unlikely to cause instability — softcap=15 and QK-norm already provide implicit regularization
- No change in speed, memory, or parameters


## Résultats
- val_bpb: 1.091936
- peak_vram_mb: 11412.5
- status: discarded
