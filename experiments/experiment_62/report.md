Add gradient clipping (max_norm=1.0) to the training loop — retry of exp58 which crashed on infra, never tested on GPU.

## What changed

Added `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` after gradient accumulation and before `optimizer.step()`. Single line addition.

## Why

Every major LLM training pipeline (GPT-2/3, LLaMA, PaLM) uses gradient clipping, but this codebase has none. The Muon optimizer has built-in normalization (Newton-Schulz orthogonalization + NorMuon step sizing), so clipping is effectively a no-op for matrix params. However, the AdamW-optimized params (token embeddings, lm_head, per-layer scalars, value embeddings) have no gradient normalization. Occasional gradient spikes can corrupt their exponential moving averages (exp_avg, exp_avg_sq), leading to suboptimal updates for many subsequent steps.

With max_norm=1.0 (industry standard), clipping only activates during extreme gradient spikes and is otherwise a complete no-op. Expected outcome:
- If neutral: gradient norms are already < 1.0, clipping never fires, identical to baseline
- If positive: occasional spikes caught, cleaner AdamW optimizer states, better convergence (~0.0005-0.002 improvement)
- If negative: max_norm=1.0 is too aggressive, clips too often, mild regression ~0.001-0.003

This was the #1 HIGH PRIORITY untested change per prompt.md notes. Exp58 attempted this but crashed on infra (never ran on GPU).

## Risks

- Zero crash/OOM risk (no new parameters, no new allocations)
- Zero speed impact (clip_grad_norm_ is a single pass over gradients, negligible vs forward/backward)
- If max_norm=1.0 is too aggressive: mild regression. If so, try max_norm=5.0 or 10.0 next.


## Résultats
- val_bpb: 1.087990
- peak_vram_mb: 10337.7
- status: discarded
