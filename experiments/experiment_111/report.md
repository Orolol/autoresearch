Per-head learned output scales in attention — learnable per-head scaling before output projection (c_proj).

## What changed

Four changes in `train.py`:

1. **CausalSelfAttention.__init__** (line 74): Added `self.head_scales = nn.Parameter(torch.ones(self.n_head))`. One learnable scalar per attention head.

2. **CausalSelfAttention.forward** (line 108): Added `y = y * self.head_scales[None, None, :, None]` before the view+c_proj. Each head's attention output is scaled by its learned factor before concatenation and output projection.

3. **GPT.init_weights**: Added `block.attn.head_scales.fill_(1.0)` to ensure proper initialization after `to_empty()`.

4. **GPT.setup_optimizer**: Separated 1D params (head_scales) from 2D params (Muon matrices) in transformer.h. Head scales are trained with AdamW at lr=scalar_lr*0.01=0.005, same conservative LR as resid_lambdas.

## Why

- **Multi-scale RoPE creates heterogeneous heads**: With bases (2500, 10000, 40000, 160000), each head specializes in a different distance range. Learned output scales let the model amplify useful heads per layer.
- **Muon can't easily adjust per-head scaling**: Newton-Schulz orthogonalization on c_proj normalizes singular values, constraining the update's ability to independently scale each head's contribution. Explicit AdamW-trained scales provide an unconstrained path.
- **Different from qk_scales** (exp68, +0.002): qk_scales adjusted attention sharpness BEFORE softmax. Head output scales adjust the contribution to the RESIDUAL STREAM after attention, which is a fundamentally different mechanism.
- **Zero cost**: 32 params total (4 per layer, 8 layers). One broadcasted multiply per layer. No measurable overhead.
- **Safe at init**: scales=1.0 means model starts at exact baseline.

## Risks

- c_proj might already learn per-head weighting → scales stay at 1.0 → neutral result
- LR=0.005 might be too low for meaningful differentiation in ~830 steps
- 32 params is so few that the effect size might be below noise threshold
- Expected range: -0.002 to +0.003

## If it works

- Try per-head-per-dimension scales (4*128=512 params/layer) for finer control
- Try higher LR for head_scales (scalar_lr * 0.05 or 0.1)
- Try initializing with different scales per head (e.g., favoring medium-range heads)
- Combine with intermediate RoPE range (1500, 8000, 60000, 300000)

## If it fails

- c_proj already handles per-head weighting. Mark head_scales as dead.
- Try intermediate RoPE range (1500, 8000, 60000, 300000)
- Try VE_LR=0.4 (between exp108's 0.2 and current 0.6)
- Try FP8 matmuls on Blackwell for 2x throughput


## Résultats
- val_bpb: 1.071584
- peak_vram_mb: 10962.8
- status: discarded
