Add gradient clipping (max_norm=1.0) to the training loop — standard LLM practice currently missing from the pipeline.

## What changed

Added `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` after gradient accumulation and before `optimizer.step()`. This is a single line addition.

## Why

Every major LLM training pipeline (GPT-2/3, LLaMA, PaLM) uses gradient clipping, but this codebase has none. The Muon optimizer has built-in normalization (Newton-Schulz orthogonalization + NorMuon step sizing), so clipping is effectively a no-op for matrix params. However, the AdamW-optimized params (token embeddings, lm_head, per-layer scalars, value embeddings) lack such normalization. Occasional gradient spikes can corrupt their exponential moving averages (exp_avg, exp_avg_sq), leading to suboptimal updates for many subsequent steps.

With max_norm=1.0 (the industry standard), clipping only activates during extreme gradient spikes and is otherwise a complete no-op. This means the expected outcome is either:
- Neutral (gradient norms are already < 1.0 → no clipping → identical to baseline)
- Positive (occasional spikes are caught → cleaner AdamW states → better convergence)

## Risks

- Zero crash/OOM risk (no new parameters, no new allocations)
- Zero speed impact (clip_grad_norm_ is a single pass over gradients, negligible vs forward/backward)
- If max_norm=1.0 is too aggressive, mild regression ~0.001-0.003
