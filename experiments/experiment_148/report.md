Sparse per-head gating on U-Net skip connections — extend the winning sparse gating pattern (exp141) to cross-layer information routing.

## What changed

Added `skip_gates`: a ModuleList of 4 Linear(8, 4) layers (one per skip connection pair) that gate the U-Net skip contributions per-head using sparse input.

In `GPT.__init__` (after skip_lambdas):
- `self.skip_gate_dim = 2 * config.n_head` (= 8, same as attention gate)
- `self.skip_gates = ModuleList([Linear(8, 4, bias=False) for _ in range(4)])`

In `GPT.forward` (skip application):
- Before: `x = x + skip_lambdas[mirror] * skip_states[mirror]`
- After: skip state is reshaped to (B, T, n_head, head_dim), gated per-head by `2*sigmoid(skip_gate(x[:, :, :8]))`, then reshaped back and scaled by skip_lambdas

In `GPT.setup_optimizer`:
- skip_gate_params added to matrix_params for Muon optimization (shape (4,8) = same group as attention gates)

Total new parameters: 4 gates x (4 x 8) = 128 params (0.0003% of model).

## Why

The sparse attention gate (exp141) was the biggest single improvement since VE bookend (-0.002318). Its success pattern: sparse input (8 dims), structured output (per-head), input-dependent, Muon-friendly matrix shape (4, 8). U-Net skip connections are another key information pathway that currently uses only scalar gating (skip_lambdas). By adding the same sparse per-head gating mechanism:

1. Different tokens can receive different amounts of early-layer information (e.g., content words may need more early features than punctuation)
2. The per-head grouping inherits structure from the attention computation in the source layer
3. The gate matrix (4, 8) is identical to the attention gate — proven Muon-optimal
4. Zero-init starts neutral (2*sigmoid(0) = 1.0), preserving current behavior

Gate_dim tuning is confirmed DONE (exp146: 4→neutral, exp147: 12→neutral, 8 is optimal). This experiment applies the optimal gate pattern to a new pathway rather than continuing to tune it.

## Risks

- **Neutral (medium-high)**: Skip states include MLP output which lacks natural head structure. The per-head grouping may be too arbitrary to benefit from gating.
- **Regression (low)**: Zero-init guarantees the model starts identical to current best. The gate can only learn to improve or stay neutral.
- **Crash (none)**: Standard linear + sigmoid ops, all shapes static, torch.compile compatible.
- Expected range: -0.003 to +0.003

## If it works

- Apply sparse gating to x0 injection (currently scalar x0_lambdas per layer)
- Try different skip gate_dim values (4, 12) — though attention gate tuning suggests 8 is robust
- Apply sparse gating to residual lambdas (per-channel, input-dependent residual scaling)

## If it fails

- Sparse gating only helps for semantically structured outputs (attention heads). Skip states lack this structure.
- Remaining directions: FP8 matmuls retry (biggest "more steps" lever), DeltaNet/GLA (complex architectural change)
- Consider that 1.064626 may be near the architecture's ceiling at this compute budget


## Résultats
- val_bpb: 1.071212
- peak_vram_mb: 12757.4
- status: discarded
