Retry torch.compile(mode="max-autotune") with cudagraph_mark_step_begin() fix to enable CUDA graphs and autotuned Triton kernels.

## What changed
- `torch.compile(model, dynamic=False)` → `torch.compile(model, dynamic=False, mode="max-autotune")`
- Added `torch.compiler.cudagraph_mark_step_begin()` before each model forward call in the training loop

## Why
- **Exp16 crashed with a clear error**: "accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run" at `x = self.transformer.wte(idx)`. The error message itself suggests the fix: call `torch.compiler.cudagraph_mark_step_begin()` before each model invocation.
- **The issue**: CUDA graphs capture and replay computation graphs. When gradient accumulation runs multiple forward passes per optimizer step, the graph from step N overwrites outputs from step N-1 before backward can use them. `cudagraph_mark_step_begin()` tells the CUDA graph runtime that a new "step" is beginning, preventing output overwriting.
- **"More steps" is the ONLY winning strategy.** MLP 3x worked by reducing compute per step. max-autotune attacks the same axis from infrastructure: CUDA graphs eliminate kernel launch overhead (~10-20μs per kernel × hundreds of kernels), and autotuned Triton kernels find optimal configurations for RTX 5090's SM architecture.
- **Zero architectural risk**: Same model, same math, same convergence properties. Only execution speed changes.
- **Compilation overhead is free**: First 10 steps are excluded from the time budget.

## Key difference from exp16 (crashed)
- Added `torch.compiler.cudagraph_mark_step_begin()` before each `model(x, y)` call in the gradient accumulation loop. This prevents CUDA graph replay from overwriting tensor outputs needed by subsequent backward passes.

## Expected outcome
- If CUDA graphs + autotuning give 5-10% throughput boost: ~40-80 extra steps → improvement of ~0.001-0.004
- If the model is already compute-bound (not launch-bound): neutral within noise (~0.003)
- Very unlikely to crash now — the fix directly addresses the exp16 error
- No OOM risk (CUDA graphs use slightly more memory for graph capture, but model is only ~10GB VRAM)
