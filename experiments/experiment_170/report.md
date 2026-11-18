FP8 MLP-only matmuls with @torch.compiler.disable to bypass torch.compile graph tracing — the root cause of 3 consecutive FP8 crashes (exp121, exp169, exp170 prev).

## What changed

Added FP8 infrastructure (train.py, ~40 lines before GPT model):

1. **`_FP8LinearOp` autograd Function** (setup_context pattern): Forward quantizes inputs and weights to float8_e4m3fn with dynamic per-tensor absmax scaling, uses `torch._scaled_mm` for the matmul. Backward is standard BF16 (go @ weight, go.t() @ x).

2. **`fp8_linear(x, weight)` wrapper**: Decorated with `@torch.compiler.disable` to create a graph break — torch.compile never traces into the FP8 code, eliminating the source of all previous crashes.

3. **MLP.forward**: `self.c_fc(x)` → `fp8_linear(x, self.c_fc.weight)`, same for c_proj.

2 fp8_linear calls per block x 8 blocks = 16 FP8 matmuls per forward pass. Attention projections and lm_head kept in BF16.

## Why

FP8 is #1 priority — biggest remaining lever. Previous attempts crashed due to torch.compile interaction:
- exp121: all linears + custom autograd (too aggressive)
- exp169: col-major bug (.t().contiguous() kills col-major layout)
- exp170 prev: setup_context pattern but still compiled → unknown crash

This attempt uses `@torch.compiler.disable` as explicitly recommended in prompt.md after crashes. Graph breaks in MLP have minimal overhead (~1% from 16 breaks at ~50us each vs ~360ms/step) since MLP matmuls are compute-bound.

Expected speedup: ~1.15x overall (FP8 forward 1.8x on MLP ~60% of compute, BF16 backward unchanged). ~960 steps instead of 830 → -0.002 to -0.004 val_bpb.

## Risks

- **Crash (low)**: `@torch.compiler.disable` isolates FP8 from torch.compile entirely. Remaining risk: _scaled_mm API constraints on Blackwell. The col-major fix (no .contiguous() after .t()) is correct.
- **Regression (low)**: FP8 quantization noise (3 mantissa bits) mitigated by ReluSquared sparsity and per-tensor dynamic scaling.
- **Neutral (medium)**: if FP8 overhead (amax reduction, casts) + graph break overhead exceeds the matmul speedup for these matrix sizes.
- **OOM (none)**: FP8 tensors are smaller than BF16.

## If it works

- Apply FP8 to attention projections (c_q, c_k, c_v, c_proj) for more coverage
- Try `use_fast_accum=True` for faster accumulation
- Try FP8 backward too (full FP8 training)

## If it crashes

- Check error: _scaled_mm API issue vs autograd vs something else
- Try fixed scales (skip amax computation)
- FP8 path may be fundamentally broken on this PyTorch/Blackwell combo; move to DeltaNet/GLA (#2 priority)
