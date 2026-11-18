Halve TOTAL_BATCH_SIZE from 2^18 to 2^17 to double optimizer steps and push the "more steps = better convergence" hypothesis further.

## What changed
- `TOTAL_BATCH_SIZE = 2**18` -> `TOTAL_BATCH_SIZE = 2**17` (single line change)
- This halves the batch size from ~262K to ~131K tokens per optimizer step
- grad_accum_steps goes from 4 to 2 (DEVICE_BATCH_SIZE=32, SEQ_LEN=2048, so tokens_per_fwdbwd=65536)
- Roughly doubles the number of optimizer steps in the 5-minute budget (~1660 vs ~830)

## Why
- The ONLY improvement that has ever beaten baseline was reducing batch size from the original to 2^18 (more optimizer steps)
- This model is severely undertrained (5 min / ~830 steps). Doubling the step count gives the optimizer 2x more parameter updates
- 131K tokens per step is still a large batch — gradient estimates should remain stable
- The overhead per optimizer step is small relative to forward/backward compute, so the wall-clock cost of 2x more steps is mostly absorbed by halving grad_accum
- Each optimizer step still processes 2 microbatches of 65K tokens, which is plenty of signal for gradient estimation
- No architectural risk — this is purely a training dynamics experiment

## Expected outcome
- If "more steps" continues to dominate: improvement of ~0.002-0.005 in val_bpb
- If gradient noise becomes limiting: regression of ~0.002-0.005 (noisier gradients from smaller batch)
- If the effects cancel: neutral result within noise (~0.003)
- No OOM risk — same DEVICE_BATCH_SIZE, just fewer accumulation steps
- No crash risk — purely a hyperparameter change


## Résultats
- val_bpb: 1.098882
- peak_vram_mb: 11412.5
- status: discarded
