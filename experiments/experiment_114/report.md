Per-dimension residual and x0 lambdas — widen from scalar per-layer to per-channel for finer residual stream routing.

## What changed

1-line change in `train.py` (GPT.__init__, line 146-147):

Before: `self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))` — shape [8]
After: `self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer, config.n_embd))` — shape [8, 512]

Before: `self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))` — shape [8]
After: `self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer, config.n_embd))` — shape [8, 512]

No other changes. The forward pass (`self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0`) works unchanged via broadcasting: indexing with `[i]` gives a [512] vector that broadcasts across [B, T, 512]. Init values (1.0 for resid, 0.1 for x0) are unchanged. Optimizer groups (AdamW) handle 2D tensors correctly.

## Why

- **Information bottleneck removal**: Scalar lambdas force all 512 dimensions to share the same residual persistence and x0 injection rate per layer. Different dimensions may encode different types of information (local vs. global features, syntactic vs. semantic) that benefit from different layer-wise dynamics.
- **Same principle as exp112 (best)**: The wider VE gate succeeded by removing a 32-dim bottleneck → full 512-dim. Here we remove a 1-dim (scalar) bottleneck → 512-dim for residual routing.
- **Multi-scale RoPE creates specialized heads**: With bases (2500, 10000, 40000, 160000), different attention heads specialize in different distance ranges. The dimensions they write to may benefit from different persistence rates.
- **Zero risk**: Exact baseline behavior at step 0. Same init values. Broadcasting handles the forward pass. No new code paths. No OOM risk (8192 params = 16KB).

## Risks

- Scalar lambdas may already capture enough → neutral result
- 8192 params with ~830 steps may not learn meaningful per-dimension differentiation
- resid_lambdas LR (0.005) might be too conservative for per-dimension learning
- Expected range: -0.003 to +0.003

## If it works

- Try per-dimension skip_lambdas too (currently shape [4], widen to [4, 512])
- Try higher resid_lambdas LR (0.01 or 0.05) for faster per-dimension differentiation
- Combine with VE weight decay or VE_LR=0.4

## If it fails

- Scalar lambdas are sufficient. Mark per-dimension lambdas as dead.
- Try FP8 matmuls on Blackwell for 2x throughput (the "more steps" lever)
- Try VE weight decay (0.02-0.04)
- Try VE_LR=0.4 (between exp108's 0.2 and current 0.6)


## Résultats
- val_bpb: 1.078830
- peak_vram_mb: 11921.9
- status: discarded
