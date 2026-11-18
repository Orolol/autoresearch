Attention gate_dim=12 (3x n_head) — retry of exp143 which crashed on infra, never GPU-tested.

## What changed

One constant change in `CausalSelfAttention.__init__()` (train.py line 77):
- `self.gate_dim = 2 * self.n_head` -> `self.gate_dim = 3 * self.n_head`
- Gate weight matrix shape changes from (n_head, 2*n_head) = (4, 8) to (n_head, 3*n_head) = (4, 12)
- All 8 layers affected. Parameter increase: 8 layers x (4x12 - 4x8) = 128 extra params (negligible).

## Why

Exp141 introduced sparse attention gating with gate_dim=8 (2x n_head), achieving -0.002318 — the biggest single improvement since VE bookend. The gate_dim was set by analogy but never tuned.

Gate_dim search status:
- gate_dim=4 (exp146): 1.064760, neutral (+0.000134 vs best 1.064626)
- gate_dim=8 (exp141): 1.064626 (current best)
- gate_dim=12 (exp143): CRASHED on infra, never tested

In modded-nanogpt PR#117, gate_dim was the critical parameter — dim=12 for 12 heads (1:1 ratio) beat larger ratios. Our 3:1 ratio (12:4) is slightly more lopsided but still well within Muon's effective range.

## Risks

- **Neutral (high)**: gate_dim=4 and 8 gave nearly identical results (0.000134 delta), suggesting the gate is robust to gate_dim. 12 will likely also be neutral, confirming 8 as optimal.
- **Regression (low)**: 3:1 aspect ratio slightly worse for Muon orthogonalization than 2:1 current.
- **Crash (none)**: Same ops, static shapes, torch.compile compatible. Identical pattern to exp141.
- Expected range: -0.002 to +0.002

## If it works

- Try gate_dim=16 (4x n_head) to see if wider input helps further
- Try gate_dim=10 to narrow the optimum between 8 and 12
- Consider per-layer gate_dim (narrow in early layers, wider in late)

## If it fails (neutral or worse)

- gate_dim=8 is confirmed optimal. Gate_dim tuning is DONE.
- Move to FP8 matmuls retry (biggest "more steps" lever) or DeltaNet/GLA (complex architectural change)
- Consider that 1.064626 may be near the architecture's ceiling at this compute budget


## Résultats
- val_bpb: 1.065164
- peak_vram_mb: 12500.0
- status: discarded
