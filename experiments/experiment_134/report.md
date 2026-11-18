Lower EMBEDDING_LR from 0.6 to 0.4 — tuning the largest untested hyperparameter controlling 22% of model parameters.

## What changed

One constant change in `train.py` (line 516):
- `EMBEDDING_LR = 0.6` → `EMBEDDING_LR = 0.4`

This affects the wte (word token embedding) learning rate in the AdamW optimizer. The effective LR after dmodel scaling is reduced from ~0.735 to ~0.490.

Zero new parameters. Zero architecture change. Zero compute overhead.

## Why

- **Never tuned**: EMBEDDING_LR=0.6 was inherited from the original setup. Despite controlling 16.7M parameters (22% of the model — the single largest parameter tensor), it has never been tested against alternatives. Every other major hyperparameter has been systematically explored.
- **High effective LR**: After dmodel scaling (×1.225 for 512-dim model), the actual LR is ~0.735. This is very high for sparse embedding updates where each token's embedding only gets gradient when that token appears in the batch.
- **VE regularization analogy**: Value Embeddings (also an embedding table) benefited from weight decay regularization (exp115, -0.00088). This suggests embedding parameters prefer conservative updates rather than aggressive ones.
- **Stabilizes rare token embeddings**: With vocab_size=32768, many rare tokens get infrequent gradient updates. A lower LR prevents these from overshooting when they do get updated.
- **Consistently recommended**: Listed as "next step" in exp132 and exp133 reports — the most commonly suggested untested change across recent experiments.

## Risks

- **Neutral (medium)**: 0.6 might already be near-optimal. The original tuning may have been done offline.
- **Regression (low)**: With only ~830 training steps, embeddings need to converge fast. Too-low LR could slow convergence for common tokens that have ample gradients.
- **Crash (none)**: Pure constant change.
- Expected range: -0.003 to +0.005

## If it works

- Try EMBEDDING_LR=0.2 (even more conservative) to find the optimal point.
- Also try EMBEDDING_LR=0.5 to narrow the range.
- Consider separate LR for VE embeddings (currently same as wte).

## If it fails

- Try EMBEDDING_LR=0.8 (higher) — maybe embeddings need even more aggressive updates in ~830 steps.
- Intermediate RoPE range (1500, 8000, 60000, 300000) — explicitly listed as untested in prompt.md.
- FP8 matmuls retry with MLP-only approach (avoid custom autograd).
- DeltaNet/GLA remains the only untried high-potential architectural change.


## Résultats
- val_bpb: 1.067798
- peak_vram_mb: 11984.2
- status: discarded
