Widen VE gate input from 32 channels to full n_embd (512) for richer per-head gating information.

## What changed

Two changes in `train.py` (CausalSelfAttention class):

1. **__init__** (line 74): Removed `self.ve_gate_channels = 32`. Changed `nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)` to `nn.Linear(self.n_embd, self.n_kv_head, bias=False)`. Gate input widens from 32 to 512 dimensions.

2. **forward** (line 85): Changed `self.ve_gate(x[..., :self.ve_gate_channels])` to `self.ve_gate(x)`. Gate now sees the full hidden state instead of the first 32 channels.

No other changes. Init (zeros), optimizer (Muon for 2D matrices), and all other code unchanged.

## Why

- **VE bookend is the biggest win** (-0.004, exp84). The gate is the control mechanism for VE — improving it has the highest EV of any remaining change.
- **32 dims = 6.25% of representation**: The first 32 dimensions of the hidden state are arbitrary. In a transformer, information is distributed across all dimensions via random initialization and training. Restricting the gate to 6.25% of the representation is a genuine information bottleneck.
- **Multi-scale RoPE needs differentiated gating**: With bases (2500, 10000, 40000, 160000), each head specializes in different distance ranges. The gate should make head-specific VE decisions based on the full token context, not just 32 dims.
- **Muon orthogonalizes gate rows**: The (4, 512) gate weight is trained by Muon (Newton-Schulz), which orthogonalizes the 4 rows. This means each head's gate response is maximally different — complementing multi-scale RoPE.
- **Zero cost at init**: Gate weight initialized to zeros → sigmoid(0)*2 = 1.0 → exact baseline behavior at step 0.
- **Negligible overhead**: 3840 extra params total (2 VE layers × 1920 params). No measurable compute difference.

## Risks

- Gate might already work well with 32 dims → widening is neutral
- Muon on (4, 512) could overfit the gate (unlikely with orthogonalization)
- Expected range: -0.003 to +0.002

## If it works

- Try adding bias to VE gate for asymmetric default gating
- Try wider gate with bottleneck: Linear(n_embd, 32) → ReLU → Linear(32, n_kv_head) for non-linear gating
- Try per-position gate (make gate depend on position as well as content)

## If it fails

- 32 channels is sufficient for VE gating. Mark wider VE gate as dead.
- Try VE_LR=0.4 (between exp108's 0.2 and current 0.6)
- Try intermediate RoPE (1500, 8000, 60000, 300000)
- Try FP8 matmuls on Blackwell for 2x throughput


## Résultats
- val_bpb: 1.070199
- peak_vram_mb: 10449.4
- status: IMPROVED
