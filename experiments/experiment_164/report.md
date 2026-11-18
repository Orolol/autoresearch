torch.compile(mode="reduce-overhead") for CUDA graph optimization to increase training throughput and get more steps.

## What changed

In `train.py` (line 629):
- Changed `torch.compile(model, dynamic=False)` to `torch.compile(model, dynamic=False, mode="reduce-overhead")`

Total new parameters: 0. Architecture: unchanged. Compute per step: unchanged. Expected throughput improvement: 10-30% (more steps in same 5-minute budget).

## Why

The model currently uses torch.compile with default mode. Each training step involves many small operations per layer: RMSNorm (x2), token shift (x2 concat + sigmoid + mix), Q/K/V projections, RoPE application, QK-norm, dynamic temperature (sigmoid + projection), SDPA or block-diagonal attention, sparse gate (projection + sigmoid), output projection, MLP (projection + ReluSquared + projection). Across 8 layers, that's 100+ kernel launches per forward pass.

With default compile mode, each kernel launch has Python dispatch overhead. `reduce-overhead` mode uses CUDA graphs to capture the entire kernel sequence and replay it in a single GPU submission, eliminating per-kernel Python overhead.

This mode was NEVER tested:
- exp16/17: `max-autotune` = CUDA graphs + Triton autotuning → CRASH (likely Triton autotuning issue)
- exp21: `max-autotune-no-cudagraphs` = Triton autotuning only → +0.009 (autotuning hurts)
- `reduce-overhead` = CUDA graphs WITHOUT Triton autotuning → UNTESTED

The model is designed to be CUDA-graph compatible:
- `torch.compile(dynamic=False)` ensures static shapes
- `gc.disable()` prevents GC-triggered memory changes
- `expandable_segments:True` prevents memory fragmentation
- All tensor shapes are constants (no data-dependent shapes)

The "more steps" lever has been the most consistent source of improvement: MLP 3x (exp14, -0.008) and local attention (exp149, -0.0015) both won by giving more training steps in the same time budget.

## Risks

- **Crash (medium)**: CUDA graphs require stable memory addresses between replays. The Muon optimizer's `torch.stack(params)` and `unbind` operations create temporary tensors. However, the optimizer functions have their own `@torch.compile(dynamic=False, fullgraph=True)` so they're compiled separately from the model.
- **Neutral (low)**: If the model is already compute-bound (high MFU), reducing launch overhead won't help. But typical MFU for small models with many operations is 30-40%, suggesting significant overhead room.
- **OOM (low)**: CUDA graphs pre-allocate replay buffers, potentially increasing peak VRAM by a few hundred MB (well within 32GB budget).
- **Regression (none)**: Same model, same math. If it runs, it's strictly faster.

Expected range: -0.010 to crash (no middle ground — either it works and gives more steps, or it crashes).

## If it works

- This would be a major breakthrough — "free" speedup from better GPU utilization
- Combine with other compute-saving techniques for multiplicative gains
- Try `mode="max-autotune"` again (CUDA graphs + Triton autotuning) — if CUDA graphs work, the exp16 crash may have been from a different issue
- The additional steps could make previously-neutral architectural changes significant

## If it fails (crash)

- CUDA graphs are incompatible with some aspect of the training pipeline
- The model is already at its compile optimization ceiling with default mode
- FP8 matmuls remain the only "more steps" lever, requiring torchao or a custom approach
- Focus shifts to DeltaNet/GLA as the last high-potential architectural direction
- Consider that 1.062436 may be near the practical ceiling for this model size and training budget
