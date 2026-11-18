Cross-layer MLP residual connection — direct skip from each layer's MLP output to the next layer's MLP input for progressive MLP computation.

## What changed

In `GPT.__init__` (train.py):
- Added `self.mlp_skip_lambdas = nn.Parameter(torch.zeros(config.n_layer))` — per-layer scalar controlling how much previous MLP output feeds into current MLP input.

In `GPT.init_weights` (train.py):
- Init `mlp_skip_lambdas` to 0.05 (matching skip_lambdas pattern for immediate activation).

In `GPT.num_scaling_params` (train.py):
- Added mlp_skip_lambdas to scalar count.

In `GPT.setup_optimizer` (train.py):
- Added mlp_skip_params group with AdamW (scalar_lr, betas=(0.96, 0.95)).
- Updated parameter assertion.

In `Block.forward` (train.py):
- Added `mlp_skip` parameter — the scaled previous MLP output.
- MLP input is now `norm(x) + mlp_skip` instead of just `norm(x)`.
- Returns tuple `(x, mlp_out)` instead of just `x`.

In `GPT.forward` (train.py):
- Track `mlp_prev = torch.zeros_like(x)` (starts at zero).
- Each layer computes `mlp_skip = self.mlp_skip_lambdas[i] * mlp_prev`.
- Block returns `(x, mlp_prev)` where mlp_prev is the current layer's MLP output.

Total new parameters: 8 scalars (one per layer). Zero additional compute beyond 8 scalar multiplies and additions.

## Why

The model has several cross-layer information pathways:
- **Residual stream**: carries all sublayer outputs, normalized before each consumer
- **U-Net skips**: early layer outputs → mirror late layer inputs (learned lambdas)
- **x0 lambdas**: initial embedding → each layer's input

Missing: **direct MLP-to-MLP communication**. Each MLP's output enters the residual stream where it gets diluted by attention outputs, x0 mixing, U-Net skips, and normalization before reaching the next MLP.

The cross-layer MLP skip creates a separate "MLP stream" — a direct path from one MLP to the next. This is particularly valuable with non-uniform MLPs (1.5x early, 4.5x late):
- Layers 0-3 (narrow 1.5x MLPs) extract basic features
- These features skip directly to layers 4-7 (wide 4.5x MLPs) without residual dilution
- Wide late MLPs can build on narrow early MLP features more efficiently

Analogy: U-Net skips connect FULL layer outputs across mirror pairs. MLP skips connect JUST MLP outputs across adjacent layers. Both create shortcuts through the network, but MLP skips are MLP-specific and sequential.

**Different from dead ideas:**
- Sparse MLP gate (exp142, +0.004): gated MLP OUTPUT per group. This connects MLP OUTPUT → next MLP INPUT.
- Soft MoD (exp116, +0.012): per-position routing (skip entire layers). This is per-LAYER weighting of MLP residual.
- Cross-block linear recurrence (exp152, +0.006): connected attention states across local blocks. This connects MLP states across layers.

## Risks

- **Neutral (medium)**: The residual stream already carries MLP output to the next layer via `x = x + mlp_out`. The direct skip bypasses normalization and attention mixing, but this additional signal may be redundant.
- **Regression (very low)**: Lambda=0.05 × mlp_prev≈0 (c_proj zero-init) = near-zero at init. Model starts near-identical to current best.
- **Crash (none)**: Standard add/multiply, all shapes static, torch.compile compatible.
- **OOM (none)**: One extra (B, T, n_embd) tensor stored = ~64MB.
- **Gradient chain (very low)**: mlp_prev creates gradient path through all previous MLPs. But lambdas are small (0.05) so gradient magnitude decays exponentially: 0.05^8 ≈ 4e-11 for full-chain gradients.

Expected range: -0.003 to +0.003.

## If it works

- Try init=0.1 or 0.2 for stronger MLP skip signal
- Try non-uniform lambdas: higher for early→late transitions (e.g., layer 3→4 where MLP width jumps from 1.5x to 4.5x)
- Try MLP skip spanning 2 layers (layer i MLP → layer i+2 MLP) for wider receptive field
- Try combining MLP skip with a small learned projection (Linear(n_embd, n_embd)) for richer feature transformation

## If it fails

- MLP-to-MLP communication is redundant with residual stream at this scale
- The normalization between layers is not actually diluting useful MLP features
- Next directions: FP8 matmuls (biggest "more steps" lever), DeltaNet/GLA (linear attention replacement)
- Consider that 1.062436 may be near the architecture's ceiling at ~830 steps


## Résultats
- val_bpb: nan
- peak_vram_mb: 13015.3
- status: discarded
