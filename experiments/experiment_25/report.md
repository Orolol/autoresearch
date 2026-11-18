Replace ReluSquared MLP with SwiGLU at iso-compute (2x hidden dim = same FLOPs/params as 3x ReluSquared).

## What changed
- MLP activation: `F.relu(x).square()` replaced by fused SwiGLU: `F.silu(gate) * up`
- c_fc shape: (1536, 512) -> (2048, 512) — fused gate+up projection
- c_proj shape: (512, 1536) -> (512, 1024) — down projection from hidden=1024
- Forward: `c_fc(x)` -> chunk into gate (1024) and up (1024) -> `silu(gate) * up` -> `c_proj`
- 4 lines changed in MLP class

## Why
- **Iso-compute swap**: 3 x 512 x 1024 = 2 x 512 x 1536 = 1,572,864 params/layer. Same FLOPs. Same total params. Only the architecture differs.
- **SwiGLU is the standard in ALL modern LLMs** (LLaMA 1/2/3, Mistral, Gemma, Qwen, DeepSeek). It provides richer multiplicative interactions via gating that ReluSquared doesn't have.
- **Previous SwiGLU failure was at 4x MLP** (old exp1/exp7, ~1.099-1.105), which was before MLP 3x was the baseline, before z-loss, before VE-last-only. The compute regime is fundamentally different now.
- **Fused gate+up is one large matmul** (512->2048) instead of two smaller ones, which is good for GPU utilization. torch.compile should handle SiLU + multiply efficiently.
- **Zero risk to step count**: same compute per step, same memory footprint, same grad_accum. The only variable is quality per step.

## Muon grouping analysis
- Old: (1536, 512) x 8 layers, (512, 1536) x 8 layers = 2 Muon groups
- New: (2048, 512) x 8 layers, (512, 1024) x 8 layers = 2 Muon groups
- NS orthogonalization cost is identical (512x512 inner products in both cases)

## Init analysis
- c_fc (fused gate+up): uniform(-s, s) — both gate and up halves initialized the same way
- c_proj (down): zeros — MLP output starts at zero (standard residual init)
- SiLU(0-centered input) * (0-centered input) ≈ 0 at init — stable

## Expected outcome
- If SwiGLU's gating provides better feature selection than ReluSquared: improvement of ~0.001-0.005
- If SwiGLU and ReluSquared are equivalent at this scale: neutral within noise (~0.003)
- If 1024 hidden is too narrow vs 1536 despite gating: mild regression (~0.001-0.003)
- No crash, OOM, speed, or memory risk — pure iso-compute architecture swap
