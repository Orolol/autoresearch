FP8 MLP-only matmuls via torch._scaled_mm for ~2x MLP throughput on Blackwell FP8 tensor cores.

## What changed

Added FP8 infrastructure and applied it to MLP only (NOT attention projections):

1. **`_FP8Linear` autograd function** (new, ~35 lines): Custom forward/backward that quantizes activations and weights to float8_e4m3fn with dynamic per-tensor absmax scaling, then uses `torch._scaled_mm` for the matmul. Backward uses float8_e5m2 for gradients (wider range). Decorated with `@torch.compiler.allow_in_graph` to treat it as an opaque op in torch.compile's graph (avoids tracing issues that caused exp121 crash).

2. **`fp8_linear(x, weight)` helper**: Reshapes 3D input to 2D, calls `_FP8Linear.apply`, reshapes back. Drop-in replacement for `F.linear(x, weight)`.

3. **MLP.forward**: `self.c_fc(x)` → `fp8_linear(x, self.c_fc.weight)`, `self.c_proj(x)` → `fp8_linear(x, self.c_proj.weight)`.

Total: 2 fp8_linear calls per block × 8 blocks = 16 FP8 matmuls per forward pass. Attention projections and lm_head kept in BF16.

## Why

This is the #1 priority item: FP8 matmuls are the biggest remaining lever for improvement. The MLP is ~60% of per-layer compute. Blackwell FP8 tensor cores run at 2x BF16 throughput. If MLP gets ~2x speedup, overall throughput increases ~1.3x → ~1080 steps instead of ~830.

Key differences from exp121 (which crashed):
- **MLP-only** instead of all linear layers (6 per block → 2 per block)
- **`@torch.compiler.allow_in_graph`** instead of raw custom autograd — tells torch.compile to treat the function as a leaf op, avoiding graph tracing issues
- **Cleaner implementation** — single class, no helper wrappers

## Risks

- **Crash (medium-high)**: torch._scaled_mm + custom autograd + torch.compile is the same pattern that crashed exp121. The `allow_in_graph` decorator is the mitigation, but it may not be sufficient.
- **Regression (low)**: FP8 quantization noise (3 mantissa bits vs 7 for bf16) could hurt MLP precision. Mitigated by: (a) ReluSquared creates many exact zeros, (b) per-tensor dynamic scaling maximizes FP8 range usage, (c) MLP is less precision-sensitive than attention.
- **Neutral (low)**: If amax reduction overhead + FP8 cast overhead exceeds the matmul speedup. Unlikely for matrices this large (768×512 and 2304×512).
- **OOM (none)**: FP8 tensors are smaller than BF16.

## If it works

- Apply FP8 to attention projections too (c_q, c_k, c_v, c_proj) for full coverage
- Try `use_fast_accum=True` in `_scaled_mm` for even faster but less precise accumulation
- Try delayed scaling (use previous step's amax) to eliminate per-call amax reduction
- The additional steps could make previously-neutral architectural changes significant

## If it crashes

- Check the error message: is it `_scaled_mm` API, torch.compile tracing, or backward computation?
- Try without `@torch.compiler.allow_in_graph` (let torch.compile trace through it)
- Try `setup_context` pattern instead of old-style `forward(ctx, ...)`
- Try wrapping MLP forward in `@torch.compiler.disable` (creates graph break but might work)
- Try fixed scale factors (skip amax) to simplify the graph
- Last resort: WD=0.3 (priority #3, safe fallback)
