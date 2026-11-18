Adaptive x0 injection — input-dependent per-position gating on x0_lambdas using sparse gating pattern for content-dependent embedding signal routing.

## What changed

In `GPT.__init__` (train.py):
- Added `self.x0_gates = nn.ModuleList([nn.Linear(8, 1, bias=False) for _ in range(n_layer)])` — per-layer sparse gate reading first 8 dims of residual stream, outputting 1 scalar per position.

In `GPT.init_weights` (train.py):
- Zero-init x0_gate weights → sigmoid(0)=0.5 → 2*0.5=1.0 (neutral, identical to current behavior at start).

In `GPT.num_scaling_params` (train.py):
- Added x0_gates parameter counting.

In `GPT.setup_optimizer` (train.py):
- Added x0_gate_params as AdamW group with scalar_lr, betas=(0.96, 0.95).
- Updated parameter assertion.

In `GPT.forward` (train.py):
- Before each layer: `x0_g = 2 * sigmoid(x0_gates[i](x[:, :, :8]))` → (B, T, 1).
- Modified: `x = resid_lambdas[i] * x + x0_lambdas[i] * x0_g * x0` (was: `x0_lambdas[i] * x0`).
- Gate reads from residual stream BEFORE resid/x0 mixing, so each layer's gate sees the previous block's output.

Total new parameters: 8 layers × (1 × 8) = 64. Compute change: negligible (8 tiny matmuls + sigmoids per forward).

## Why

The x0_lambdas inject raw embedding (x0) into each layer's input — critical for training (removing them = +0.020). Currently, this injection is a FIXED scalar per layer: every position gets the same strength regardless of token content or context.

Different positions likely have different needs:
- **Rare tokens**: weak representation in residual stream → more raw embedding helps downstream layers
- **Common tokens**: strong residual representation → less raw embedding needed
- **Near BOS**: less accumulated context → more embedding signal useful
- **Deep in document**: rich context → embedding signal less important

The sparse gating pattern (gate_dim=8 → sigmoid → scalar) has produced the two most recent wins:
- Attention gate (exp141, -0.002318): per-head attention output gating
- Dynamic temperature (exp153, -0.000644): per-head Q scaling

Both use `Linear(8, output, bias=False)` → `2 * sigmoid` → modulate a critical operation. This experiment applies the SAME pattern to x0 injection.

**Different from dead ideas:**
- Per-dimension lambdas (exp114, +0.009): 512 per-CHANNEL lambdas without content-dependence. This is 1 per-POSITION scalar with content-dependence.
- Soft MoD (exp116, +0.012): per-position routing to skip entire layers. This only modulates x0 contribution.

## Risks

- **Neutral (medium-high)**: Optimal x0 strength may be position-independent — the scalar x0_lambda may already be optimal for all positions. The gate could learn nothing useful.
- **Regression (very low)**: Zero-init starts identical to current best. 64 params is negligible overhead.
- **Crash (none)**: Standard Linear + sigmoid, all shapes static, torch.compile compatible.
- **OOM (none)**: 64 params, negligible memory.

Expected range: -0.002 to +0.002.

## If it works

- Try per-channel x0 gating: Linear(8, n_embd) instead of Linear(8, 1) for finer control
- Try adaptive resid_lambdas gating (modulate residual stream retention per position)
- Try shared x0_gate across layers (fewer params, potentially better generalization)
- Try larger gate_dim (16, 32) for richer input signal

## If it fails

- x0 injection is inherently position-independent at this scale
- The sparse gating pattern works for attention mechanisms but not for residual stream operations
- Next directions: FP8 matmuls (biggest "more steps" lever), DeltaNet/GLA (full attention replacement)
- Consider that 1.062436 may be near the architecture's practical ceiling at ~830 steps


## Résultats
- val_bpb: 1.078273
- peak_vram_mb: 12632.3
- status: discarded
