Sparse attention gate — per-head output gating using only the first 8 dims of input, inspired by modded-nanogpt PR#117 world record.

## What changed

Added a sparse attention gate to `CausalSelfAttention` in `train.py`:

1. **New parameter**: `self.attn_gate = nn.Linear(gate_dim, n_head, bias=False)` where `gate_dim = 2 * n_head = 8`. Weight shape is (4, 8) — near-square, well-suited for Muon optimizer.

2. **Gate application**: After SDPA output (before c_proj), compute `gate = 2 * sigmoid(attn_gate(x[:, :, :8]))` and multiply per-head attention output by the gate. Applied to all 8 layers.

3. **Initialization**: Gate weights zero-init, so `2 * sigmoid(0) = 1.0` — neutral at start (same pattern as VE gate).

4. **Parameter overhead**: 8 layers x 4 x 8 = 256 params (negligible vs ~46M total).

## Why

- **Modded-nanogpt evidence**: PR#117 set a world record (-55 steps) with a sparse attention gate. The key finding: `dim=12` (sparse) substantially outperformed `dim=96` and `dim=768` (dense). The reason: the gate matrix maps a vector to independent per-head scalars. With a lopsided shape like (768, 6), Muon's Newton-Schulz orthogonalization doesn't work well. A near-square shape like (12, 6) works much better.

- **Exp131 was not a fair test**: Exp131 tried attention output gating with full n_embd=512 input → (512, 4) weight matrix. This is extremely lopsided (128:1 ratio). The sparse variant with (4, 8) has a 2:1 ratio, matching the optimal ratio from the PR.

- **Sparsity matches**: 8/512 = 1.6% of input dims, same ratio as 12/768 in the original PR.

- **Mechanism**: The gate lets each position decide per-head how much to trust the attention output. Heads that found relevant keys get amplified; heads that didn't find useful patterns get suppressed. This is input-dependent (query-aligned) and position-specific.

## Risks

- **Neutral (medium)**: Exp131 was +0.0007 with dense gate. The sparse trick may not transfer from 124M to ~46M model scale.
- **Regression (low)**: Gate could suppress useful attention. But zero-init starts neutral, and 2*sigmoid range (0, 2) allows both suppression and amplification.
- **Crash (none)**: Standard ops, static shapes, torch.compile compatible.
- Expected range: -0.003 to +0.003

## If it works

- Try different gate_dim values (4, 12, 16) to find the optimum
- Try adding a sparse gate on MLP output as well (modded-nanogpt also explored this)
- Try gating with bias (learned offset from 1.0)

## If it fails

- Attention gating axis is definitively dead (both dense and sparse variants tested)
- Remaining high-potential ideas: FP8 matmuls (retry with compile-friendly approach), DeltaNet/GLA (complex but untested)
- Consider that 1.067 may be near the capacity limit for this architecture at this compute budget


## Résultats
- val_bpb: 1.064626
- peak_vram_mb: 12499.2
- status: IMPROVED
