Differential Attention (DiffTransformer) with proper V-halving for noise cancellation and ~12% parameter savings.

## What changed

Changes in `CausalSelfAttention` and `GPT`:

1. **DiffAttn sub-heads** (`CausalSelfAttention.__init__`): Added `sub_head_dim = head_dim // 2 = 64`. Q/K projections unchanged (same weights, reshaped as 8 sub-heads of dim 64 instead of 4 heads of dim 128). V projection halved: `c_v` outputs `n_kv_head * sub_head_dim = 256` instead of 512. `c_proj` input halved: 256->512 instead of 512->512. Added `diff_lambda = 0.5`.

2. **Attention forward** (`CausalSelfAttention.forward`): Q/K reshaped to `(B, T, 2*n_head, sub_head_dim)`. V reshaped to `(B, T, n_kv_head, sub_head_dim)`. VE applied at sub_head_dim. For SDPA, V doubled via `repeat_interleave(2, dim=1)` so each sub-head pair shares the same V. After SDPA, output reshaped to `(B, T, n_head, 2, sub_head_dim)` and paired sub-heads subtracted: `y[:,:,:,0] - 0.5 * y[:,:,:,1]`. Result: `(B, T, n_head * sub_head_dim)` = `(B, T, 256)` fed into halved c_proj.

3. **Value embeddings** (`GPT.__init__`): `kv_dim = n_kv_head * sub_head_dim` (halved). VE embedding size goes from (vocab, 512) to (vocab, 256).

4. **RoPE** (`GPT.__init__`, `init_weights`): Bases duplicated for sub-head pairs: `(2500,2500,10000,10000,40000,40000,160000,160000)`. Precomputed at `sub_head_dim=64` instead of `head_dim=128`.

## Why

- **DiffTransformer** (Microsoft, 2024): subtracting paired attention outputs cancels noise in attention patterns while preserving signal. Both sub-heads attend to identical V, so the subtraction isolates differential attention signal.
- **Compute savings**: halved V, c_proj, and VE save ~12% total params, translating to ~12% more training steps (~925 vs ~830). MLP 3x (exp14, -0.008) showed that more steps is the single most important lever.
- **QK unchanged**: Q/K projections are identical weights just reshaped — QK attention FLOPs unchanged.
- Exp99 tried a simpler DiffAttn (reshape only, no V halving) but crashed on infra. This is the first proper DiffAttn test.

## Risks

- HEAD_DIM//2=64 as standalone was +0.012 worse (exp61), but DiffAttn subtraction is mechanistically different
- Halved V (128->64 per head) reduces value information capacity
- Fixed lambda=0.5 may not be optimal (paper uses learnable ~0.8)
- Expected range: -0.008 to +0.012

## If it works

- Try learnable lambda (per-head, per-layer) instead of fixed 0.5
- Try lambda=0.8 (closer to paper's preferred value)
- Try DiffAttn on only deep layers (keep standard attention for early layers)
- Combine with sandwich norm (exp100-104, never tested on GPU)

## If it fails

- V capacity loss from 128->64 outweighs noise cancellation + step gains
- Try DiffAttn WITHOUT V halving: keep full V, split post-SDPA (exp99's approach but working)
- Try wider multi-scale RoPE (1000, 5000, 50000, 500000)
- Focus on Schedule-Free Adam or FP8 matmuls


## Résultats
- val_bpb: 1.079623
- peak_vram_mb: 10607.7
- status: discarded
