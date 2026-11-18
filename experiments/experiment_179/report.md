Chunkwise linear attention with learned per-head decay replacing block-diagonal softmax SDPA in local layers (2-5) — priority #1 DeltaNet/GLA direction.

## What changed

In `train.py`, replaced the block-diagonal softmax attention (SDPA) in local layers (2-5) with chunkwise linear attention using a learnable per-head exponential decay:

1. **New parameter** (`CausalSelfAttention.__init__`, local layers only):
   - `decay_rate`: per-head scalar (shape `(n_head,)` = 4 params per layer, 16 total)
   - `gamma = sigmoid(decay_rate)` controls the exponential decay rate

2. **Replaced forward** (`CausalSelfAttention.forward`, local_attn branch):
   - **Before**: Block-diagonal SDPA on 256-token blocks (no cross-block context)
   - **After**: Chunkwise linear attention with 256-token chunks:
     - **Intra-chunk**: `(Q @ K^T / sqrt(D)) * decay_causal_mask @ V` — causal linear attention with exponential decay `gamma^(i-j)` replacing softmax normalization
     - **Cross-chunk**: `Q @ S * gamma^(pos+1)` — queries the recurrent KV state accumulated from all previous chunks
     - **State update**: `S = gamma^C * S + K_weighted^T @ V / sqrt(D)` — exponential decay prevents noise accumulation

3. **Init** (`GPT.init_weights`):
   - `decay_rate = 4.0` → `gamma = sigmoid(4.0) ≈ 0.982` → effective context ~55 tokens
   - Learnable: model can adjust to shorter (faster decay) or longer (slower decay) context per head

Total new parameters: 16 scalars (4 heads x 4 local layers). Minimal compute overhead (~1-2% from cross-chunk matmuls).

## Why

1. **Priority #1 on the queue**: DeltaNet/GLA is the top remaining direction — "Remplacer softmax attention. Potentiel le plus élevé."
2. **Full-sequence context**: Local layers currently see only 256 tokens (block-diagonal). Linear attention with recurrent state gives layers 2-5 access to the entire 2048-token sequence.
3. **Different from addons**: exp152 (+0.006) and exp177 (+0.004) tried ADDING cross-block context ON TOP of softmax. This REPLACES softmax entirely — simpler optimization, single mechanism.
4. **Compute-neutral**: Intra-chunk linear attention has similar FLOPs to SDPA (256x256 matmul). Cross-chunk overhead is minimal (D^2 per position vs D*C for SDPA).
5. **Exponential decay**: Prevents the noise accumulation that plagued exp152 (no decay) while being simpler than DeltaNet's delta rule.

## Key design choices

- **No feature map**: Raw signed dot products (QK^T/sqrt(D)). With QK-norm + zero-init c_proj, the model learns appropriate scaling. Simpler than ELU+1 or ReLU feature maps.
- **Chunk size = 256**: Same as previous block size. 8 chunks per sequence (fixed loop, compile-friendly).
- **Float32 decay masks**: Computed in float32 for numerical stability with large exponents (gamma^255), then cast to bf16.
- **Per-head learnable decay**: Different heads can specialize in different context lengths.

## Risks

- **Regression (high)**: Removing softmax from 4 layers is aggressive. exp178 (sigmoid attention, +0.020) showed that non-softmax attention mechanisms can regress significantly. Linear attention is different from sigmoid but shares the "no normalization" property.
- **Crash (low-medium)**: The 8-iteration loop with state should compile fine, but never tested with torch.compile.
- **OOM (none)**: State S is (B, H, D, D) = 32 * 4 * 128 * 128 * 2 bytes = 4.2MB. Negligible.
- **Compile time (medium)**: The loop will be unrolled into 8 sequential operations, potentially increasing compile time.


## Résultats (after crash fix)
- val_bpb: 1.050748
- peak_vram_mb: 14120.4
- status: IMPROVED
