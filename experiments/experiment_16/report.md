Enable torch.compile(mode="max-autotune") for CUDA graphs and autotuned Triton kernels to increase training throughput.

## What changed
- `torch.compile(model, dynamic=False)` → `torch.compile(model, dynamic=False, mode="max-autotune")`
- Single line change, no architectural modifications

## Why
- **"More steps" is the ONLY winning strategy.** Exp14 (MLP 3x) proved that reducing compute per step to get more optimizer steps is the dominant factor. This experiment attacks the same axis from the infrastructure side.
- **max-autotune enables two key optimizations:**
  1. **CUDA graphs**: Captures the entire forward/backward as a graph, eliminating per-kernel launch overhead (~10-20μs per kernel × hundreds of kernels = significant CPU overhead). Especially impactful for small models where CPU overhead is a larger fraction of total step time.
  2. **Autotuned Triton kernels**: Tries multiple kernel configurations during compilation to find the fastest one for each operation. Default mode uses heuristics; max-autotune benchmarks alternatives.
- **Zero architectural risk**: Same model, same math, same convergence properties. Only the execution speed changes.
- **Compilation overhead doesn't matter**: First 10 steps are excluded from the time budget (`if step > 10`), so extra autotuning time during compilation is free.
- **Blackwell (RTX 5090) benefits**: Modern GPU with high kernel launch rates, but CUDA graphs still eliminate overhead entirely. The autotuned kernels may find better tile sizes for Blackwell's SM architecture.

## Expected outcome
- If CUDA graphs + autotuning give 5-10% throughput boost: ~40-80 extra steps → improvement of ~0.001-0.004
- If the model is already compute-bound (not launch-bound): neutral within noise (~0.003)
- If CUDA graphs have compatibility issues with the training loop: potential crash (but unlikely with `dynamic=False` and a simple loop)
- No OOM risk (CUDA graphs use slightly more memory for graph capture, but model is only ~10GB VRAM)
