State-augmented hybrid attention — inject 16 virtual memory tokens from recurrent state S into intra-chunk softmax for unified local+cross-chunk attention routing.

## What changed

In `train.py`, modified the chunkwise hybrid attention in local layers (2-5) to include virtual memory tokens in the softmax attention:

1. **New parameter** (`CausalSelfAttention.__init__`, local layers only):
   - `n_mem = 16` — number of virtual memory tokens per head
   - `mem_read`: Parameter of shape (n_head * n_mem, head_dim) = (64, 128) — learned read patterns for extracting virtual KV from recurrent state. Stored as 2D for proper Muon optimization.

2. **Modified forward** (`CausalSelfAttention.forward`, local_attn branch):
   - Before the chunk loop: precompute augmented causal mask (BS, M+BS) where first M columns are always-True (memory visible from all positions) and last BS columns follow the original causal mask. Precompute augmented positional bias with zero bias for memory tokens and log-decay for real tokens.
   - Inside each chunk iteration:
     - `mem_k = mem_read.view(H, M, D).expand(B, ...)` — learned memory keys (same for all chunks)
     - `virtual_v = mem_k @ S` — read values from recurrent state using learned patterns
     - Concatenate `[mem_k, kc]` and `[virtual_v, vc]` to form augmented KV
     - Softmax attention over (BS, M+BS) positions: memory tokens compete with local tokens for attention weight
     - **Kept** the linear cross-chunk term `Q @ S * cross_decay` as complementary smooth pathway
   - State update (decay + accumulate) is **unchanged**

3. **Init** (`GPT.init_weights`):
   - `mem_read` initialized with uniform(-s, s), same scale as attention weights

## Why

1. **Priority queue explicitly lists this as untested**: "state-augmented attention (inject recurrent state as virtual memory tokens in softmax)".
2. **Structural novelty**: Last 5 experiments (190-194) were all chunk attention variations (GLA, extend layers, chunk size, multi-scale decay). This is a fundamentally different approach — changing HOW the model accesses cross-chunk context, not tweaking the mechanism.
3. **Unified attention routing**: Currently `output = softmax(local) + linear(cross)` — the model has no mechanism to choose between local and long-range. With memory tokens in softmax, the model naturally routes attention between local tokens and compressed past context through softmax competition. A query that needs distant context will attend more to memory tokens; one that needs local context will attend to real tokens.
4. **Complementary pathways**: Linear cross term provides smooth dense baseline; memory-augmented softmax provides sharp selective access. The model can use both.
5. **Minimal overhead**: 32K new params (0.07%), attention matrix 6.25% larger in 4/8 layers (~1-2% net compute).

## Risks

- **Regression (medium)**: Memory tokens are initially random — until mem_read converges, they add noise to the softmax. The linear cross term mitigates this.
- **Crash (low)**: Static shapes (272 = 256+16 constant), standard ops (cat, expand, matmul, softmax). compile-friendly.
- **Neutral (medium)**: Linear cross term may already provide adequate cross-chunk context, making memory augmentation redundant.
- **Compile overhead (low-medium)**: More ops in the chunk loop (cat, expand) may increase torch.compile time.


## Résultats (after crash fix)
- val_bpb: 1.105106
- peak_vram_mb: 26913.9
- status: discarded
