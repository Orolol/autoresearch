Soft Mixture of Depths — input-dependent per-block gating for compile-friendly token routing.

## What changed

Three changes in `train.py` (Block class + init_weights):

1. **Block.__init__** (line 127): Added `self.block_gate = nn.Linear(config.n_embd, 1, bias=False)`. One gate per block, maps hidden state to a scalar routing weight.

2. **Block.forward** (lines 131-135): Compute `nx = norm(x)`, then `gate = 2 * sigmoid(block_gate(nx))` (shape [B, T, 1]). Gate scales both sublayer outputs: `x = x + gate * attn_out` and `x = x + gate * mlp_out`. The norm(x) is reused as attention input (saves one redundant norm computation).

3. **GPT.init_weights** (line 199): Added `torch.nn.init.zeros_(block.block_gate.weight)` for zero-init → sigmoid(0)*2=1.0 → exact baseline at step 0.

Total new params: 8 layers x (1, 512) = 4096 params. Negligible (<0.01% of model). Gate weights trained by Muon (2D matrix, unique shape group (1, 512)).

## Why

- **Mixture of Depths (piste #6)** is the last major untried architectural idea. Standard MoD is incompatible with torch.compile(dynamic=False) because it requires dynamic token counts. This soft formulation avoids the issue: every token goes through every layer, but with a learned input-dependent scaling factor.
- **Tokens vary in difficulty**: Common words, punctuation, and predictable tokens don't need the full 8-layer capacity. Easy tokens can learn to "skip" layers (gate→0), freeing the model to focus capacity on harder tokens (gate→2).
- **Uses proven VE gate pattern**: The `2*sigmoid` with zero-init is the same mechanism used for VE gates (exp84 -0.004, exp112 -0.000167). It provides exact baseline behavior at step 0 and smooth learned gating during training.
- **Compile-friendly**: All shapes are static. The gate is [B, T, 1] broadcast-multiplied with [B, T, n_embd]. No conditionals, no dynamic routing. torch.compile handles this perfectly.

## Risks

- 8 layers may not have enough redundancy for routing to help — gates stay near 1.0 → neutral result (most likely)
- Muon on (1, 512) shape: Newton-Schulz degenerates to row normalization. Effective LR = MATRIX_LR = 0.04. May be too low or too noisy for gate learning.
- Gates could diverge to extreme values (0 or 2) destabilizing training — mitigated by the sigmoid bounding and Muon's normalization.
- Expected range: -0.005 to +0.005

## If it works

- Try gating only attention (not MLP) for cheaper routing
- Try per-sublayer separate gates (different gate for attn vs MLP)
- Try combining with temperature-dependent gate sharpening during warmdown
- Try initializing middle layers with slightly lower gate bias (encourage middle-layer skipping)

## If it fails

- 8 layers too few for routing benefits. Mark soft MoD as dead at this depth.
- Try VE betas (0.9, 0.95) for smoother VE updates (piste #13)
- Try wte embedding weight decay (same logic as exp115's VE WD success)
- Try FP8 matmuls on Blackwell for 2x throughput (piste #3)


## Résultats
- val_bpb: 1.081621
- peak_vram_mb: 11473.4
- status: discarded
