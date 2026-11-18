Causal depthwise convolution before MLP for local feature mixing (Mamba/H3-inspired).

## What changed

Three changes in `Block`, `GPT.init_weights`, and `GPT.setup_optimizer`:

1. **Block.__init__**: Added `self.conv = nn.Conv1d(n_embd, n_embd, kernel_size=4, padding=0, groups=n_embd, bias=False)`. Depthwise (per-channel) causal convolution with 4-token receptive field.

2. **Block.forward**: Replaced `x = x + self.mlp(norm(x))` with:
```python
h = norm(x)
h = self.conv(F.pad(h.transpose(1, 2), (3, 0))).transpose(1, 2)
x = x + self.mlp(h)
```
The MLP now receives locally-mixed features (current + 3 previous positions) instead of pure per-position normalized features.

3. **GPT.init_weights**: Conv initialized to identity (weight[:, 0, -1] = 1.0, rest zero). At step 0, conv is a pass-through — model starts identically to baseline.

4. **GPT.setup_optimizer**: Conv params (3D, 512x1x4) separated from Muon matrix params and placed in AdamW group with LR=0.05.

## Why

- **MLP is position-independent**: After attention aggregates global context into the residual stream, the MLP processes each position independently. A short conv before MLP gives it access to raw neighboring features — complementary to attention's learned aggregation.
- **Mamba/H3 precedent**: State-of-the-art hybrid architectures (Mamba, H3, Hyena) use short causal convolutions before their core mixing operation. The conv captures low-level local patterns that attention may smooth out.
- **Negligible cost**: 16,384 extra params (0.03%), <1% wall-time overhead (memory-bound op, ~0.6ms per step on 1.8 TB/s bandwidth).
- **Information flow**: The biggest wins in this project came from information flow improvements (VE bookend -0.004, U-Net skip -0.001, skip_lambdas -0.002). This adds another pathway.
- **Safe at init**: Identity initialization means zero regression risk at step 0. The conv can only help.

## Risks

- Conv might not learn useful mixing if attention already provides sufficient local context → result equals baseline (neutral)
- AdamW LR=0.05 for conv might be too high or too low
- F.pad + transpose adds minor memory bandwidth overhead
- Expected range: -0.005 to +0.003

## If it works

- Try kernel_size=8 for wider receptive field
- Try conv before attention too (pre-attention local mixing)
- Try gated conv: `h = h * sigmoid(conv(h))` instead of direct replacement
- Combine with other untested ideas (Schedule-Free Adam, wider RoPE)

## If it fails

- Local mixing before MLP doesn't add value when attention already provides context. Mark as dead.
- Try VE LR separation (option 9: lower VE LR from 0.6 to ~0.15)
- Try Schedule-Free Adam for AdamW groups
- Try FP8 matmuls on Blackwell


## Résultats
- val_bpb: 1.093080
- peak_vram_mb: 10986.2
- status: discarded
