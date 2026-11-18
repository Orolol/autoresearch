Remove x0_lambdas (per-layer skip connection from initial embedding) to simplify the residual stream.

## What changed
- Removed `self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))` from GPT.__init__
- Removed `x0 = x` and `self.x0_lambdas[i] * x0` from the forward loop
- Each layer now receives `self.resid_lambdas[i] * x` instead of `self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0`
- Removed x0_params from optimizer setup and param counting
- No other changes to architecture or hyperparameters

## Why
- x0_lambdas (initialized to 0.1) adds 10% of the initial embedding to each layer's input, creating a skip connection from the embedding to every layer
- VE (value embeddings) already provides a direct token→value signal in the last layer's attention, which may make x0_lambdas redundant
- Removing it simplifies the forward pass: one fewer multiply+add per layer, no x0 tensor kept in memory
- This is explicitly listed as an untested simplification idea in the notes
- If quality holds, it's a clean win (less code, fewer params, slightly faster)

## Expected outcome
- If x0_lambdas is redundant with VE: neutral result (within ~0.003 noise), which would be a simplification win
- If x0_lambdas provides critical gradient flow: regression of ~0.005-0.015
- No crash/OOM risk — purely removes a component


## Résultats
- val_bpb: 1.108641
- peak_vram_mb: 10964.5
- status: discarded
