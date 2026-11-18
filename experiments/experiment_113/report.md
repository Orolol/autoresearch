Key Embedding (KE) — extend VE to also contribute to keys via a separate learned gate, reusing the shared value embedding.

## What changed

Three changes in `train.py` (CausalSelfAttention class + init_weights):

1. **__init__** (line 75): Added `self.ke_gate = nn.Linear(self.n_embd, self.n_kv_head, bias=False)` for VE layers (first and last). Same shape as ve_gate: (4, 512) = 2048 params per layer.

2. **forward** (lines 88-89): After VE is added to values, also add it to keys: `k_gate = 2 * torch.sigmoid(self.ke_gate(x))` then `k = k + k_gate.unsqueeze(-1) * ve`. Reuses the same VE embedding — no new embedding parameters. Applied before RoPE and QK-norm.

3. **init_weights** (lines 201-202): Added `torch.nn.init.zeros_(block.attn.ke_gate.weight)` for ke_gate layers. Zero-init → sigmoid(0)*2=1.0 → neutral at step 0.

Total new params: 2 layers × 2048 = 4096 params. Negligible (~0.01% of model). ke_gate trained by Muon (2D matrix, grouped with ve_gate).

## Why

- **VE is the most productive direction**: bookend (-0.004), skip_lambdas (-0.002), wider gate (-0.000167). Extending VE to keys is the natural architectural expansion.
- **Keys determine attention patterns**: Currently, what gets attended to is solely determined by the learned c_k projection. Adding token identity to keys means the model can attend based on WHAT token is present (identity), not just the learned representation. E.g., matching brackets, finding keywords.
- **Reuses existing VE embedding**: The same shared first+last layer embedding provides both value and key identity signals. Different learned gates control how much each head uses identity for V vs K.
- **Muon orthogonalizes ke_gate rows**: Newton-Schulz ensures each head's key gating is maximally different, complementing multi-scale RoPE specialization.
- **Applied before RoPE**: KE gets position-encoded, providing "token X at this position" signal rather than pure identity.

## Risks

- c_k might already encode sufficient token info → ke_gate stays near 0 → neutral
- Could make attention too identity-focused (copying) at expense of contextual patterns
- Expected range: -0.004 to +0.003

## If it works

- Try different ke_gate init (e.g., 0.1 for faster activation)
- Try KE on query as well (QE): token identity in queries for "what am I looking for"
- Try separate KE embedding (not shared with VE) for independent key identity
- Combine with VE weight decay for regularization

## If it fails

- Token identity in keys is redundant with c_k. Mark KE as dead.
- Try VE weight decay (0.02-0.04) for VE regularization
- Try VE_LR=0.4 (between exp108's 0.2 and current 0.6)
- Try FP8 matmuls on Blackwell for 2x throughput


## Résultats
- val_bpb: 1.070299
- peak_vram_mb: 10450.4
- status: discarded
