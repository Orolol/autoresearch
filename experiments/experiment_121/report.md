FP8 matmuls on Blackwell — replace all nn.Linear calls in MLP and attention with float8_e4m3fn matmuls via torch._scaled_mm for ~2x throughput on FP8 tensor cores.

## What changed

Added FP8 matrix multiplication infrastructure and applied it to all linear layers in the transformer:

1. **`_Float8Matmul` autograd function**: Custom forward/backward that quantizes activations and weights to `float8_e4m3fn` with dynamic per-tensor scaling, then uses `torch._scaled_mm` for the matmul. Both forward and backward matmuls run in FP8.

2. **`fp8_linear(x, weight)` helper**: Reshapes input to 2D, calls `_Float8Matmul.apply`, reshapes back. Drop-in replacement for `F.linear(x, weight)`.

3. **MLP.forward**: `self.c_fc(x)` → `fp8_linear(x, self.c_fc.weight)`, `self.c_proj(x)` → `fp8_linear(x, self.c_proj.weight)`.

4. **CausalSelfAttention.forward**: `self.c_q(x)` → `fp8_linear(x, self.c_q.weight)`, same for c_k, c_v, c_proj. VE gate (`self.ve_gate(x)`) kept in BF16 (tiny 512→4 matrix, not worth FP8).

Total: 6 fp8_linear calls per block × 8 blocks = 48 FP8 matmuls per forward pass. lm_head kept in BF16.

## Why

- All VE/hyperparameter tuning paths are exhausted (exp117-120 all neutral/discarded)
- FP8 is the biggest remaining lever: RTX 5090 Blackwell has FP8 tensor cores at 2x BF16 throughput
- MLP 3x (exp14, -0.008) proved "more steps = better val_bpb". FP8 achieves more steps via faster compute
- MLP linear ops (60% of per-layer compute) + attention projections (40%) = 100% coverage of linear compute
- Expected: ~1.3-1.5x overall throughput → 30-50% more training steps → potential -0.003 to -0.005 val_bpb

## Risks

- **Crash (medium)**: torch._scaled_mm API or torch.compile interaction might fail
- **Regression (low-medium)**: quantization overhead (per-tensor amax reductions) might negate FP8 speedup for small matrices (512×512)
- **OOM (none)**: FP8 uses less memory than BF16
- Expected range: -0.005 to +0.005 (high variance due to implementation uncertainty)

## If it works

- Also apply FP8 to lm_head (512→32768, huge matmul)
- Try delayed scaling (previous step's amax) to eliminate per-call amax reduction overhead
- Try FP8 for SDPA attention computation itself

## If it crashes

- Check torch._scaled_mm return type (tuple vs tensor)
- Check torch.compile compatibility — try removing torch.compile or using mode="default"
- Fall back to FP8 for MLP only (larger matrices, more likely to benefit)
- Try torchao Float8Linear if available in PyTorch 2.10+
