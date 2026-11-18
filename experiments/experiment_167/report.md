Learnable softcap — replace fixed softcap=13 constant with a gradient-optimized parameter via softplus(raw)+1 for continuous optimization of logit compression.

## What changed

In `GPT.__init__` (train.py):
- Added `self.softcap_raw = nn.Parameter(torch.tensor(0.0))` — raw parameter for learnable softcap.

In `GPT.init_weights` (train.py):
- Added `self.softcap_raw.fill_(12.0)` — softplus(12) + 1 ≈ 13.0 (identical to current fixed value).

In `GPT.forward` (train.py):
- Replaced `softcap = 13` with `softcap = F.softplus(self.softcap_raw) + 1.0`.
- The formula `softcap * tanh(logits / softcap)` is unchanged — only the softcap VALUE is now learned.

In `GPT.num_scaling_params` (train.py):
- Added `self.softcap_raw.numel()` to scalar count.

In `GPT.setup_optimizer` (train.py):
- Added `softcap_params` as AdamW group with lr=scalar_lr*0.1=0.05, betas=(0.96, 0.95).
- Updated parameter assertion.

Total new parameters: 1 scalar. Compute change: negligible (1 softplus + 1 add per forward pass).

## Why

The logit softcap controls the maximum logit magnitude via `cap * tanh(logits / cap)`. It was hand-tuned at only 3 integer values:
- softcap=30: +0.005 (too loose)
- softcap=15: original baseline
- softcap=13: -0.001 (current best)
- softcap=11: +0.001 (too tight)

The sensitivity curve (11 < 13 > 15) shows a clear non-monotonic optimum near 13. But the true optimum may be non-integer (e.g., 12.5, 13.7). Making the softcap learnable lets the optimizer find the exact continuous minimum via gradient descent.

The gradient through `cap * tanh(logits / cap)` wrt cap is:
- `tanh(l/c) - (l/c) * sech^2(l/c)` — well-defined and smooth
- Large when logits are near the cap value (where the cap matters most)
- Small when logits are far below the cap (where capping doesn't matter)

The softplus(raw)+1 parameterization ensures softcap is always >= 1.0, preventing division-by-zero or logit collapse. The moderate LR (0.05) allows the cap to shift by ~1-2 units over 830 steps — enough to find the optimum between 11-15.

## Risks

- **Neutral (high)**: softcap=13.0 may already be very close to the continuous optimum.
- **Regression (very low)**: init is exactly 13.0 (neutral). The LR is conservative enough to prevent large drift.
- **Crash (none)**: standard PyTorch ops, all shapes static, torch.compile compatible.
- **OOM (none)**: 1 scalar parameter.

## If it works

- The learned cap value reveals the true optimum (inspect final softcap_raw value)
- Consider per-HEAD softcap: different heads may benefit from different logit ranges
- Consider per-LAYER softcap if applied to intermediate representations
- The principle "hand-tuned constant → learned parameter" could apply to other fixed values

## If it fails

- softcap=13 is confirmed as near-optimal continuously, not just discretely
- Softcap axis is fully exhausted (3 discrete + 1 continuous test)
- Remaining directions: FP8 matmuls (compute reduction), DeltaNet/GLA (architecture)
- Model is near practical ceiling ~1.062 at ~830 steps


## Résultats
- val_bpb: 1.063240
- peak_vram_mb: 13520.5
- status: discarded
