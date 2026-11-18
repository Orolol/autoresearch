Increase DEVICE_BATCH_SIZE from 32 to 64 to halve gradient accumulation steps (4→2) and improve GPU utilization with the smaller MLP 3x model.

## What changed
- `DEVICE_BATCH_SIZE = 32` → `DEVICE_BATCH_SIZE = 64`
- Single line change

## Why
- **Grad accumulation drops from 4 to 2.** Each optimizer step currently requires 4 forward-backward micro-steps. With batch=64, only 2 micro-steps are needed. This eliminates 2 Python loop iterations per step, reducing overhead.
- **Better GPU utilization.** Larger batch sizes fill tensor cores more efficiently. With MLP 3x, matmul shapes are (B*T, 512) × (512, 1536). Doubling B from 32 to 64 doubles the M dimension, improving SM occupancy.
- **VRAM is not a concern.** MLP 3x peaked at 10.1GB with batch=32. Doubling activations should use ~14-16GB, well within the 32GB RTX 5090.
- **Previous batch=64 test was with 4x MLP** (exp1 run3: 1.103, +0.015 regression). That model was 25% larger per layer. With 3x MLP, the compute/memory profile is different — the GPU may actually benefit from the larger batch.
- **"More steps" is the only winning strategy.** This targets the same axis: less wall-clock time per optimizer step → more steps in 5 minutes.

## Expected outcome
- If reduced grad_accum overhead + better GPU utilization gives more steps: improvement of ~0.001-0.004
- If batch=64 doesn't actually speed up wall-clock per step (GPU already saturated at 32): neutral within noise (~0.003)
- If larger micro-batches hurt convergence properties somehow: mild regression (~0.002-0.005)
- No crash risk. No OOM risk (10.1GB → ~14-16GB, well within 32GB).


## Résultats
- val_bpb: 1.095282
- peak_vram_mb: 20222.2
- status: discarded
