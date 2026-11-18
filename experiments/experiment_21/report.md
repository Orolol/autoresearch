Enable torch.compile(mode="max-autotune-no-cudagraphs") for autotuned Triton kernels without CUDA graphs.

## What changed
- `torch.compile(model, dynamic=False)` → `torch.compile(model, dynamic=False, mode="max-autotune-no-cudagraphs")`
- Single line change, no architectural modifications

## Why
- **Exp16 and exp17 crashed** because `max-autotune` enables CUDA graphs, which are incompatible with gradient accumulation (multiple forward passes overwrite graph outputs). The fix (`cudagraph_mark_step_begin()`) didn't work either.
- **`max-autotune-no-cudagraphs`** was introduced in PyTorch 2.3 specifically for this scenario. It gives the autotuning benefits (benchmarks multiple Triton kernel configurations to find the fastest) WITHOUT enabling CUDA graphs.
- **Autotuned kernels may help** because the default mode uses heuristics to pick Triton tile sizes, while autotuning benchmarks alternatives. Our model has specific matmul dimensions (512×1536 MLP, 512×32768 lm_head) where the default tile sizes may not be optimal for RTX 5090's SM architecture.
- **"More steps" is the ONLY winning strategy.** MLP 3x (-0.008) proved that faster execution → more steps → better convergence. Autotuned kernels attack the same axis from the infrastructure side.
- **Zero architectural risk**: Same model, same math, same convergence properties. Only kernel selection changes.
- **Compilation overhead is free**: First 10 steps are excluded from the time budget. Autotuning happens during compilation, adding maybe 30-60 seconds to the warmup phase.

## Expected outcome
- If autotuned kernels give 3-7% throughput boost: ~25-60 extra steps → improvement of ~0.001-0.003
- If the default heuristics already pick optimal kernels for these shapes: neutral within noise (~0.003)
- If autotuning overhead is excessive (unlikely with only ~5 unique matmul shapes): slight regression from slower compilation eating into effective training time
- No crash risk (no CUDA graphs). No OOM risk. No quality risk.


## Résultats
- val_bpb: 1.098823
- peak_vram_mb: 11264.4
- status: discarded
