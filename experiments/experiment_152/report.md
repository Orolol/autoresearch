Cross-block linear attention recurrence for local layers — propagate KV state between blocks for long-range context at <2% compute cost.

## What changed

In `CausalSelfAttention.__init__` (train.py):
- Added `self.recur_gate = nn.Parameter(torch.empty(self.n_head))` for local attention layers only (layers 2-5)

In `CausalSelfAttention.forward`, local_attn branch (train.py):
- After block-diagonal SDPA, compute cross-block linear attention recurrence:
  1. Per-block KV state: `delta_S = einsum('bnshd,bnshe->bnhde', k_b, v_b)` — shape (B, n_blocks, H, D, D)
  2. Causal cumsum + shift right: block i receives accumulated state from blocks 0..i-1
  3. Query recurrent state: `recur_out = einsum('bnshd,bnhde->bnshe', q_b, S_prev)`
  4. Normalize recurrent output (RMS norm per-position-per-head)
  5. Add gated recurrent output: `y = y + sigmoid(recur_gate) * norm(recur_out)`

In `GPT.init_weights` (train.py):
- Initialize `recur_gate` to -2.0 (sigmoid(-2) = 0.12, small initial contribution)

Total new parameters: 4 local layers x 4 heads = 16 scalars (0.00004% of model).

## Why

Block-diagonal local attention (exp149, -0.001546) has hard block boundaries: position 256 cannot see position 255. This is the architecture's main weakness after exp150 (layer selection done) and exp151 (block_size tuning done).

The cross-block recurrence adds a linear attention "memory" that lets each block query a compressed summary of all previous blocks. This is the same principle as RetNet/RWKV/TransNormer: efficient long-range context via state passing.

Key insight: the recurrence adds <2% total compute because:
- Local SDPA compute is only 1.3% of total per-token FLOPs
- MLP (34%) and lm_head (42%) dominate compute
- The recurrence (2 einsums per local layer) roughly doubles local attention compute = +1.3%

The design is safe:
- RMS normalization ensures consistent scale regardless of how many blocks contributed
- Learnable gate starts at 0.12 (sigmoid(-2)), preserving current behavior at init
- The model can learn to ignore recurrence (gate → 0) if it doesn't help

## Risks

- **Neutral (medium)**: Full-attention layers (0,1,6,7) already provide global context. The recurrence may be redundant.
- **Regression (low)**: Gate starts small, model begins nearly identical to current best.
- **Crash (low)**: Standard ops (einsum, cumsum, cat, norm). All shapes static.
- **Torch.compile (low-medium)**: einsum is well-supported by inductor. The (B, NB, H, D, D) tensor is 33.5 MB — should fit easily.
- **OOM (low)**: ~536 MB additional memory for backprop intermediates. Current peak 12.2 GB on 32 GB GPU.

Expected range: -0.003 to +0.005.

## If it works

- Try different gate init values (-4 for more conservative, 0 for equal mixing)
- Try applying recurrence to only the 2 deepest local layers (4-5) where context matters most
- Try a low-rank recurrence (project K to R=32 dims before state) for even less compute
- Try removing full attention from layers 1 and 6 (since recurrence now provides cross-block context)

## If it fails

- Cross-block context is already sufficiently provided by full-attention layers and token shift
- Remaining directions: FP8 matmuls (biggest "more steps" lever), DeltaNet/GLA (full linear attention replacement)
- Consider that 1.063080 may be near the architecture's ceiling at this compute budget


## Résultats
- val_bpb: 1.068952
- peak_vram_mb: 13654.5
- status: discarded
