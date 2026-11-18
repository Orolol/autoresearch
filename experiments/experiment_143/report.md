Attention gate_dim tuning: increase from 8 to 12 (3x n_head) for wider sparse input to per-head output gate.

## What changed

One constant change in `CausalSelfAttention.__init__()` (train.py line 77):
- `self.gate_dim = 2 * self.n_head` -> `self.gate_dim = 3 * self.n_head`

This changes the sparse attention gate input from 8 dims to 12 dims. The gate weight matrix shape goes from (4, 8) to (4, 12). All 8 layers are affected.

Parameter impact: 8 layers x (4x12 - 4x8) = 128 extra params (negligible vs ~46M total).

## Why

- **Direct follow-up from exp141**: The sparse attention gate was the biggest single improvement in recent history (-0.002318, new best 1.064626). Both exp141 and exp142 reports explicitly recommend gate_dim tuning as the next step.
- **Exp142 (MLP gating) failed**: +0.004 regression. MLP output groups lack the semantic structure of attention heads. This redirects focus back to optimizing the attention gate.
- **Modded-nanogpt evidence**: In PR#117, gate_dim was the critical tuning knob. dim=12 substantially beat dim=96 and dim=768 for a 6-head model. Our gate_dim=8 was chosen by 2x n_head analogy but never tuned.
- **More input information**: 12 dims (2.3% of 512) gives the gate 50% more features to base per-head decisions on. The gate maps from a sparse input signal to n_head=4 outputs — more input dims = richer per-head modulation.
- **Still Muon-friendly**: (4, 12) matrix has 3:1 aspect ratio, still well within Muon's effective orthogonalization range (exp131's failure was at 128:1 ratio).

## Risks

- **Neutral (medium-high)**: gate_dim=8 may already be optimal for our 4-head model. The improvement from 8→12 could be within the ~0.003 noise band.
- **Regression (low)**: Slightly more lopsided matrix (3:1 vs 2:1) could reduce Muon orthogonalization quality. But the difference is small.
- **Crash (none)**: Same ops, static shapes, torch.compile compatible. Only the Linear layer width changes.
- Expected range: -0.003 to +0.003

## If it works

- Try gate_dim=16 (4x n_head) to see if wider input helps further
- Try gate_dim=4 (1x n_head, perfectly square) to test the "Muon-friendliness vs info" tradeoff
- Consider per-layer gate_dim (narrow in early layers, wider in late layers)

## If it fails

- Try gate_dim=4 (square matrix, most Muon-friendly) — the 3:1 ratio may be too lopsided
- If both 4 and 12 fail: gate_dim=8 is the optimum, attention gate tuning is done
- Remaining directions: FP8 matmuls (retry with compile-friendly approach), DeltaNet/GLA (complex but untested)
