Affine RMSNorm — add learnable per-channel gain parameters to pre-sublayer norms and final norm.

## What changed

Added learnable gain vectors (`nn.Parameter`) to all RMSNorm applications, turning parameterless `F.rms_norm(x, (x.size(-1),))` into affine `F.rms_norm(x, (x.size(-1),), weight=gamma)`.

1. **Block.__init__**: Added `self.attn_norm_weight` and `self.mlp_norm_weight` (both `nn.Parameter(torch.empty(n_embd))`).

2. **Block.forward**: Changed from `norm(x)` to `F.rms_norm(x, (x.size(-1),), weight=self.attn_norm_weight)` before attention, and similarly for MLP.

3. **GPT.__init__**: Added `self.final_norm_weight` (`nn.Parameter(torch.empty(n_embd))`) for the final norm before lm_head.

4. **GPT.forward**: Changed final `norm(x)` to `F.rms_norm(x, (x.size(-1),), weight=self.final_norm_weight)`.

5. **init_weights**: All norm gains initialized to 1.0 (neutral — model starts identical to current parameterless norm).

6. **setup_optimizer**: Block norm weights automatically captured by `time_mix_params` (1D params in transformer.h, scalar_lr with betas=(0.96, 0.95)). `final_norm_weight` added as separate AdamW group with same settings. Assertion and `num_scaling_params` updated.

Total new params: 17 × 512 = 8,704 (0.02% of model). Zero compute overhead.

Note: QK norm (`norm(q), norm(k)` in attention) and post-embedding norm (`norm(x)` after wte) remain parameterless — these serve different purposes (attention stabilization and embedding scale normalization respectively).

## Why

- Every major LLM (LLaMA, Mistral, Gemma) uses affine RMSNorm. This model's parameterless norm was a simplification that may sacrifice per-layer feature selection.
- The model already benefits from per-layer specialization (non-uniform MLP 1.5x/4.5x, multi-scale RoPE, U-Net skip, resid/x0/skip lambdas). Affine norm adds a new specialization axis: each sublayer can learn to amplify different feature dimensions.
- All other architectural avenues exhausted: MLP asymmetry (exp130), token shift (exp126-128), attention output gating (exp131), VE tuning done.

## Next steps if this works

- Try different LR for norm gains (currently uses scalar_lr=0.5 via time_mix group). Lower LR might be more stable.
- Try affine norm on QK norm too (currently parameterless).

## Next steps if this fails

- FP8 matmuls retry with different approach (MLP-only, or wait for torchao in deps)
- Try removing the post-embedding norm entirely (let raw embeddings flow into residual stream)
- Consider EMBEDDING_LR tuning (0.6 was never optimized — largest untested hyperparameter)


## Résultats
- val_bpb: 1.069988
- peak_vram_mb: 11983.4
- status: discarded
