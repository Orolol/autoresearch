Sparse MLP gate — per-group output gating using first 8 dims of input, extending exp141's successful sparse attention gate to MLP output.

## What changed

Added a sparse MLP output gate to `Block` in `train.py`:

1. **New parameter**: `self.mlp_gate = nn.Linear(gate_dim, n_groups, bias=False)` where `gate_dim = 2 * n_head = 8` and `n_groups = n_head = 4`. Weight shape is (4, 8) — identical to the attention gate, near-square for Muon.

2. **Gate application**: After MLP forward, reshape output into 4 groups of 128 dims, compute `gate = 2 * sigmoid(mlp_gate(norm_x[:, :, :8]))` and multiply per-group. Gate reads from the post-attention normalized input, so it can condition on what attention just computed.

3. **Initialization**: Gate weights zero-init, so `2 * sigmoid(0) = 1.0` — neutral at start (same pattern as attention gate and VE gate).

4. **Parameter overhead**: 8 layers x 4 x 8 = 256 params (negligible vs ~46M total).

## Why

- **Direct follow-up from exp141**: Sparse attention gate was the biggest improvement in recent history (-0.002318, new best 1.064626). Report explicitly recommends trying MLP output gating next.
- **Same proven architecture**: Identical gate_dim=8, n_outputs=4, (4,8) matrix shape. Muon handles this well.
- **Modded-nanogpt evidence**: PR#117 explored both attention and MLP gating. Both contributed to the world record.
- **Different signal**: MLP gate reads from post-attention normalized input, while attention gate reads from pre-attention input. The MLP gate can condition on attention's computation to decide which MLP output channels to amplify or suppress.
- **Groups = soft channel routing**: The 4 groups of 128 dims give the MLP a soft mixture-of-experts flavor without routing overhead or dynamic shapes.

## Risks

- **Neutral (medium)**: MLP output lacks the natural head structure of attention. The 4 groups are arbitrary dimension splits — they may not have meaningful semantic differences worth gating.
- **Regression (low)**: Gate could suppress useful channels. But zero-init starts neutral, and the (0, 2) range allows both suppression and amplification.
- **Crash (none)**: Standard ops, static shapes, torch.compile compatible. Identical pattern to attention gate which compiled fine.
- Expected range: -0.003 to +0.003

## If it works

- Try different n_groups values (2, 8, 16) to find the optimum
- Try gate_dim=4 or 12 for different Muon aspect ratios
- Combine with attention gate_dim tuning (try 4, 12, 16 for attention too)
- Consider per-layer gate_dim (smaller in early layers, larger in late)

## If it fails

- MLP gating is dead — MLP output doesn't benefit from per-group gating
- Focus on attention gate_dim tuning (try 4, 12, 16 for the attention gate only)
- Consider FP8 matmuls retry with a simpler compile-friendly approach
- DeltaNet/GLA remains the last untested high-potential architectural direction


## Résultats
- val_bpb: 1.068535
- peak_vram_mb: 13015.4
- status: discarded
