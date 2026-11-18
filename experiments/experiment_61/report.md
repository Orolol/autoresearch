Reduce HEAD_DIM from 128 to 64 to double the number of attention heads (4 to 8) at zero parameter/compute cost.

## What changed

Single constant change: `HEAD_DIM = 128` -> `HEAD_DIM = 64` (line 451 of train.py).

With model_dim=512:
- Before: 4 heads x 128 dim each
- After: 8 heads x 64 dim each

## Why

1. **More diverse attention patterns.** With only 4 heads, each must multiplex many attention behaviors (local syntax, long-range semantics, positional patterns). With 8 heads, each can specialize, improving overall attention quality.

2. **Parameter-neutral.** Q, K, V projections are all 512->512 regardless of head count. Total parameter count is identical.

3. **Compute-neutral.** SDPA cost is proportional to n_embd x seq_len, not affected by how heads are split. Training speed (steps/sec) should be identical.

4. **Well-validated dimension.** GPT-2-small used 12 heads x 64 dim for 768 dim. HEAD_DIM=64 is the most common choice for small transformers. RoPE with 64-dim heads is standard.

5. **Never tested in this series.** All 60 prior experiments kept HEAD_DIM=128. This is a structural architectural change (priority #1).

## Risks

- If 128-dim heads are better with QK-norm (larger normalization space): mild regression ~0.001-0.003
- Attention is slightly sharper with smaller d (1/sqrt(64) = 0.125 vs 1/sqrt(128) = 0.088 scaling after QK-norm)
- No crash, OOM, or speed risk — pure constant change

## Expected outcome

- If head diversity helps at this scale: improvement ~0.0005-0.002 (val_bpb ~1.077-1.078)
- If neutral: within noise (~0.001)
- If 128-dim heads were better: mild regression ~0.001-0.003


## Résultats
- val_bpb: 1.090947
- peak_vram_mb: 10357.2
- status: discarded
