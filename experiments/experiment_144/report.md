Attention gate_dim tuning: reduce from 8 to 4 (1×n_head) for perfectly square Muon-optimal gate matrix.

## What changed

One constant change in `CausalSelfAttention.__init__()` (train.py line 77):
- `self.gate_dim = 2 * self.n_head` → `self.gate_dim = self.n_head`
- Gate matrix shape changes from (n_head, 2*n_head) = (4, 8) to (n_head, n_head) = (4, 4)

## Why

Exp141 introduced sparse attention gating with gate_dim=8 (2×n_head), achieving -0.002318 — the biggest single improvement since VE bookend (exp84). The gate_dim parameter was set by analogy with modded-nanogpt PR#117 but never tuned for our model.

Muon's Newton-Schulz polar decomposition converges in 5 iterations for any matrix, but is theoretically optimal on square matrices where the singular value distribution is most uniform. A 4×4 matrix (gate_dim=4) is perfectly square, vs the current 4×8 (2:1 aspect ratio).

In modded-nanogpt PR#117, gate_dim was the critical parameter — smaller values (12 for a 12-head model, i.e., 1:1 ratio) substantially outperformed larger ones. Our 4-head model with gate_dim=4 would be the exact 1:1 ratio.

Exp143 tried gate_dim=12 (3:1 ratio) but crashed on infra. Testing the opposite direction bounds the search space: if 4 works, optimal is ≤8; if it fails, optimal is ≥8.

## Expected outcome

- If better (-0.001 to -0.003): gate_dim=4 is the new optimum. Try gate_dim=2 or 3 next.
- If neutral (±0.001): gate_dim=4 and 8 bracket the optimum. Retry 12 or try 6.
- If worse (+0.002+): 4 dims insufficient, gate_dim=8 may be optimal. Retry 12 to confirm.
