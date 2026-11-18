Replace ReluSquared MLP with SwiGLU at iso-compute (2x hidden dim with fused gate+up = same FLOPs/params as 3x ReluSquared). 3rd attempt — exp25/exp28 both crashed on RunPod infra, never tested on GPU.

## What changed
- MLP activation: `F.relu(x).square()` replaced by fused SwiGLU: `F.silu(gate) * up`
- c_fc shape: (1536, 512) -> (2048, 512) — fused gate+up projection
- c_proj shape: (512, 1536) -> (1024, 512) — down projection from hidden=1024
- Forward: `c_fc(x)` -> chunk into gate (1024) and up (1024) -> `silu(gate) * up` -> `c_proj`
- 4 lines changed in MLP class, zero changes elsewhere

## Why
- **Iso-compute swap**: 1,572,864 params/layer for both. FLOPs/layer identical. Same memory. Only architecture differs.
- **SwiGLU is the standard in ALL modern LLMs** (LLaMA 1/2/3, Mistral, Gemma, Qwen, DeepSeek). Multiplicative gating provides richer feature interactions than ReluSquared's element-wise squaring.
- **Never tested on GPU** — exp25 and exp28 both crashed due to RunPod infra (CUDA driver init failure, API timeout). Code was verified correct each time.
- **Infra is now stable** — exp29 (EMA) ran successfully to completion. No reason SwiGLU should crash.
- **Previous SwiGLU failure was at 4x MLP** in the old experiment series (~1.099-1.105), before MLP 3x baseline, z-loss, VE-last-only. The compute regime is fundamentally different now.
- **Fused gate+up is one large matmul** (512->2048) — good for GPU utilization. torch.compile handles SiLU + multiply efficiently.

## Muon grouping analysis
- Old: (1536, 512) x 8 layers, (512, 1536) x 8 layers = 2 Muon groups
- New: (2048, 512) x 8 layers, (1024, 512) x 8 layers = 2 Muon groups
- NS orthogonalization cost identical (512x512 inner products in both cases)

## Init analysis
- c_fc (fused gate+up): uniform(-s, s) — both gate and up halves initialized identically
- c_proj (down): zeros — MLP output starts at zero (standard residual init)
- SiLU(0-centered) * (0-centered) ≈ 0 at init — stable, same as ReluSquared at init

## Expected outcome
- If SwiGLU gating > ReluSquared squaring: improvement of ~0.001-0.005
- If equivalent at this scale: neutral within noise (~0.003)
- If 1024 hidden too narrow despite gating: mild regression (~0.001-0.003)
- No crash, OOM, speed, or memory risk — pure iso-compute architecture swap


## Résultats
- val_bpb: 1.083872
- peak_vram_mb: 10332.4
- status: discarded
