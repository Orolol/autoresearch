Remove post-embedding RMSNorm — let wte embeddings retain magnitude information instead of forcing unit RMS.

## What changed

One-line removal in `GPT.forward()` (line 335):
- Removed `x = norm(x)` after `x = self.transformer.wte(idx)`.

Before:
```python
x = self.transformer.wte(idx)
x = norm(x)  # forces RMS=1 for all token embeddings
x0 = x
```

After:
```python
x = self.transformer.wte(idx)
x0 = x
```

Zero new parameters. Net code removal (simplification).

## Why

- **Frees embedding expressivity**: RMSNorm forces all token embeddings to unit RMS, discarding magnitude information. Without it, the model can learn different scales for different tokens — common tokens might develop larger embeddings ("louder" signal), rare tokens smaller ones.
- **Richer x0 residual**: x0 is injected at every layer via x0_lambdas. Currently x0 has fixed RMS=1. Without norm, x0 carries richer, token-dependent magnitude information throughout all 8 layers.
- **Training stability preserved**: Pre-sublayer norms (before attention and MLP in each Block) still normalize the residual stream input. The post-embedding norm was an extra constraint, not a safety mechanism.
- **Near no-op at init**: wte is initialized with std=1.0. For 512 dimensions, the RMS is approximately 1.0 by CLT, so the norm barely changes values at initialization.
- **Standard practice**: Many successful models (GPT-2, GPT-3, OPT) don't normalize after embedding. Gemma uses a scale factor (sqrt(d_model)) rather than normalization.
- **Suggested next step**: Listed in exp132 report's "next steps if this fails."

## Risks

- **Neutral (high)**: Embedding magnitudes might naturally stay near unit RMS due to AdamW optimizer dynamics. In that case, removing the norm changes nothing.
- **Regression (low)**: If some embeddings grow large during training, x0_lambdas would inject large values at every layer. But x0_lambdas starts at 0.1 and is learned, so it can adapt.
- **Crash (none)**: Pure code removal.
- Expected range: -0.003 to +0.003

## If it works

- Embedding magnitude is a useful signal. Try adding a learnable embedding scale factor: `x = wte(idx) * self.emb_scale`.
- Investigate which tokens develop the largest/smallest embeddings.
- Try removing other norms that might be redundant.

## If it fails

- Post-embedding norm is beneficial — the constraint is helpful, not harmful.
- Try EMBEDDING_LR tuning (0.6 was never optimized, largest untested hyperparameter).
- FP8 matmuls retry with MLP-only approach (avoid custom autograd, use torch._scaled_mm directly).
- Intermediate RoPE range (1500, 8000, 60000, 300000) — explicitly listed as untested.


## Résultats
- val_bpb: 1.067579
- peak_vram_mb: 11728.0
- status: discarded
