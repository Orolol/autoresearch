Attention gate_dim=4 (1x n_head, perfectly square Muon matrix) — retry of exp144 which crashed on infra, never tested on GPU.

## What changed

One constant change in `CausalSelfAttention.__init__()` (train.py line 77):
- `self.gate_dim = 2 * self.n_head` -> `self.gate_dim = self.n_head`
- Gate weight matrix shape changes from (n_head, 2*n_head) = (4, 8) to (n_head, n_head) = (4, 4)
- All 8 layers affected. Parameter reduction: 8 layers x (4x8 - 4x4) = 128 fewer params (negligible).

## Why

Exp141 introduced sparse attention gating with gate_dim=8 (2x n_head), achieving -0.002318 — the biggest single improvement since VE bookend (exp84). The gate_dim was set by analogy but never tuned.

Key arguments for gate_dim=4:
1. **Muon-optimal**: Newton-Schulz polar decomposition is theoretically optimal on square matrices. A (4,4) matrix has 1:1 aspect ratio vs (4,8) current 2:1.
2. **Modded-nanogpt evidence**: In PR#117, gate_dim=n_head (1:1 ratio, dim=12 for 12 heads) substantially outperformed larger ratios.
3. **Never tested**: Exp144 tried this exact change but crashed on infra (not code). Exp143 (gate_dim=12, 3:1) also crashed. Neither result is known.
4. **Bounds the search**: If 4 works -> try 2 or 3. If 4 fails -> 8 is optimal, retry 12.

## Risks

- **Neutral (medium-high)**: 4 dims = 0.8% of 512. May be too sparse for useful per-head gating decisions.
- **Regression (low)**: With insufficient input signal, gate converges to near-uniform -> effectively removing it -> lose exp141's gains. But zero-init starts neutral, so even a useless gate shouldn't regress.
- **Crash (none)**: Same ops, static shapes, torch.compile compatible. Identical pattern to exp141.
- Expected range: -0.003 to +0.003

## If it works

- Try gate_dim=3 or 2 to push the "less is more" hypothesis
- Try gate_dim=6 as middle ground between 4 and 8
- Consider per-layer gate_dim (square in early layers, wider in late)

## If it fails

- Try gate_dim=12 (retry exp143) to test the other direction
- If both 4 and 12 fail: gate_dim=8 is confirmed optimal, gate tuning done
- Move to FP8 matmuls (last "more steps" lever) or DeltaNet/GLA (complex architectural change)


## Résultats
- val_bpb: 1.064760
- peak_vram_mb: 12499.2
- status: discarded
