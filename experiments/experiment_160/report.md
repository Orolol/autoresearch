Learned absolute position embeddings added to token embeddings, complementing RoPE with explicit absolute position for MLP and residual stream.

## What changed

In `GPT.__init__` (train.py):
- Added `self.pos_embd = nn.Parameter(torch.zeros(config.sequence_len, config.n_embd))` — a (2048, 512) learned parameter encoding absolute position.

In `GPT.init_weights` (train.py):
- Cast pos_embd to bf16 (matching wte and VE).
- Zero-init preserves identical behavior to current best at start.

In `GPT.forward` (train.py):
- Changed `x = self.transformer.wte(idx)` to `x = self.transformer.wte(idx) + self.pos_embd[:T]`
- Position embedding is added BEFORE norm, so it affects the normalized x0 signal that propagates to all layers via x0_lambdas.

In `GPT.estimate_flops` (train.py):
- Added pos_embd to nparams_exclude (not a matmul parameter).

In `GPT.num_scaling_params` (train.py):
- Added pos_embd as a separate parameter category.

In `GPT.setup_optimizer` (train.py):
- Added pos_embd to its own AdamW group with embedding_lr, same betas as wte.
- Updated parameter assertion to include pos_embd.

Total new parameters: 2048 x 512 = 1,048,576 (1.5% of total model). Compute change: negligible (one vector addition per forward pass).

## Why

The model currently has NO absolute position encoding:
- **RoPE**: encodes RELATIVE position, but only in Q/K (attention-side only)
- **Token shift**: 1-step temporal context for K/V (local only)
- **MLP**: completely position-blind — processes every position identically
- **Residual stream**: carries no explicit position signal
- **x0_lambdas**: inject initial embedding at every layer, but initial embedding has no position info

Adding learned absolute position embeddings creates a genuinely new information pathway:
1. **MLP gets position info**: can learn position-dependent feature transformations (e.g., "near BOS = less context = use different processing")
2. **x0 propagation**: position info reaches ALL layers through x0_lambdas without per-layer PE
3. **Hybrid RoPE + learned PE**: RoPE handles relative position in attention, learned PE handles absolute position everywhere. This combination has not been tested in this codebase.
4. **Document structure**: with BOS-aligned packing, position correlates with document boundaries. Position 0 = right after BOS. High positions = deep in document with rich context.

GPT-2 used learned position embeddings as its ONLY position encoding. Later models replaced them with RoPE. But the combination (RoPE + learned PE) is novel here and could be additive: RoPE for relative attention patterns, learned PE for absolute context.

## Risks

- **Neutral (medium-high)**: MLP may not benefit from absolute position info at 2048 context — the residual stream already carries position info from attention outputs (which used RoPE). The marginal value of explicit position may be low.
- **Interference (low)**: Absolute PE contributes to Q/K through linear projections, on top of RoPE rotation. Could create conflicting signals, but the model can learn to project out the absolute component if unhelpful.
- **Regression (very low)**: Zero-init means the model starts identical to exp153 (current best).
- **Crash (none)**: Standard nn.Parameter + addition, all shapes static, torch.compile compatible.
- **OOM (none)**: 1.05M params + 2.1M optimizer state = ~6.3MB. Negligible vs 12GB peak.

Expected range: -0.003 to +0.003.

## If it works

- Try position embedding with weight decay (regularize position representations)
- Try smaller pos_embd (e.g., 256-dim with up-projection to 512) for fewer parameters
- Try per-layer position embeddings instead of shared (different position info per layer)
- Try sinusoidal position encoding (fixed, no parameters) to isolate "position info helps" vs "learned PE helps"

## If it fails

- Absolute position info is redundant with RoPE + attention residual stream at this scale
- The model is near its architecture ceiling at ~830 steps
- Next directions: FP8 matmuls (biggest "more steps" lever), DeltaNet/GLA (full attention replacement)
- Consider that 1.062436 may be the practical limit for this model size and training budget


## Résultats
- val_bpb: 1.066460
- peak_vram_mb: 12510.9
- status: discarded
