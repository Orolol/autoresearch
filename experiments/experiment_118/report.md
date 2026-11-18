WTE-as-value identity for all layers — inject normalized wte embedding (x0) directly into attention values via per-head learnable scalars, extending VE's token identity concept to middle layers at near-zero cost.

## What changed

Five changes in `train.py`:

1. **GPT.__init__** (line 149): Added `self.wte_v_scales = nn.Parameter(torch.zeros(config.n_layer, config.n_kv_head))` — shape [8, 4] = 32 learnable scalars controlling per-layer per-head wte identity injection into values.

2. **CausalSelfAttention.forward** (lines 88-91): After VE gate block, added wte identity injection: `x0_v = x0.view(B, T, n_kv_head, head_dim)` then `v = v + wte_v_scale * x0_v`. Applies to ALL layers (VE and non-VE). The normalized wte embedding (x0) is reshaped to value dimensions and scaled by per-head learned scalars.

3. **Block.forward** and **CausalSelfAttention.forward** signatures: Added `x0` and `wte_v_scale` parameters, passed from GPT.forward through the block chain.

4. **GPT.setup_optimizer**: Added `wte_v_params = [self.wte_v_scales]` as a new AdamW group with lr=scalar_lr=0.5, betas=(0.96, 0.95), same as skip_lambdas. Updated parameter assertion.

5. **GPT.init_weights**: Added `self.wte_v_scales.fill_(0.0)`. Also updated `estimate_flops` and `num_scaling_params` to include wte_v_scales.

Total new params: 32 (8 layers x 4 heads). Negligible (~0.00004% of model).

## Why

- **VE is the most productive direction**: bookend (-0.004), skip_lambdas (-0.002), wider gate (-0.000167), VE WD (-0.00088). Extending token identity in values is the highest-EV remaining change.
- **Middle layers lack token identity in values**: Layers 1-6 currently compute values purely from c_v(residual_stream). The raw "what token is this" signal is diluted through multiple layers of processing. Direct wte injection preserves token identity for value computation.
- **Different from exp85 (VE all layers, +0.010)**: That experiment added a full 16.8M-param shared VE embedding to all layers, likely overfitting. This uses the existing wte embedding with only 32 scalar params — >500,000x lighter.
- **Zero overhead**: x0 is already computed (norm(wte(idx))). No extra embedding lookups, no extra memory, negligible extra FLOPs (one reshape + multiply + add per layer).
- **Per-head specialization**: Multi-scale RoPE creates heads with different distance specializations. Local-focused heads (base 2500) may want less token identity while long-range heads (base 160000) may benefit from more.

## Risks

- c_v already accesses wte identity through the residual stream → scales stay near 0 → neutral (most likely)
- Adding to VE layers (0, 7) could interfere with VE gating, but zero-init prevents this at start
- torch.compile: condition `if x0 is not None` is always True during training, clean trace
- Expected range: -0.003 to +0.003

## If it works

- Try input-dependent gating (nn.Linear instead of scalar) for richer wte identity control
- Try wte identity in keys as well (KE was neutral in exp113, but with lighter-weight scalars it might differ)
- Try different LR for wte_v_scales (higher for faster learning, lower for stability)
- Try wte_v_scales only for non-VE layers (exclude layers 0, 7)

## If it fails

- Direct wte identity in values is redundant with residual stream. Mark wte-in-values as dead.
- Try VE WD=0.005 (stronger regularization, follow-up on exp115)
- Try VE betas (0.9, 0.95) for smoother VE updates (piste #13)
- Try FP8 matmuls on Blackwell for 2x throughput (piste #3)


## Résultats
- val_bpb: 1.082322
- peak_vram_mb: 10449.4
- status: discarded
