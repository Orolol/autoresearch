RWKV-style input-dependent per-head attention output gating on all layers.

## What changed

Three surgical changes in `CausalSelfAttention`:

1. **`__init__`** (line 76): Added `self.output_gate = nn.Linear(self.n_embd, self.n_head, bias=False)` — one per attention layer.

2. **`forward`** (lines 117-119): Before c_proj, gate each head's output:
   - `gate = 2 * sigmoid(self.output_gate(x))` — (B, T, n_head), input-dependent
   - `y = y * gate.unsqueeze(-1)` — per-head scaling of attention output
   - At init: 2*sigmoid(0) = 1.0 → neutral (model starts identical to current)

3. **`init_weights`**: Added `torch.nn.init.zeros_(block.attn.output_gate.weight)` for neutral initialization.

Total new params: 4 × 512 × 8 = 16,384 (0.02% of model). Negligible compute cost.

## Why

- **VE gating analogy**: The identical mechanism (2*sigmoid, Linear→n_head, zero-init) was the biggest single win in recent history (exp84, -0.004). That gates VE contribution to values on 2 layers. This extends the principle to gate the entire attention output on ALL 8 layers.
- **Input-dependent vs static**: Per-head STATIC scales (exp111, +0.001) were dead because c_proj already handles static per-head mixing. Input-dependent gating is fundamentally different — 2048 params per layer producing unique gates for every token.
- **RWKV/Mamba pattern**: Output gating is a core component of RWKV (output gate in WKV) and Mamba (selective gating). These architectures use content-dependent gates to control what information passes to the residual stream.
- **Dynamic head specialization**: The gate lets the model suppress irrelevant heads per-token (e.g., head specialized for long-range patterns when processing local bigrams) and amplify relevant ones.

## Risks

- **Neutral (medium)**: c_proj already provides implicit per-head weighting. The gate may be redundant.
- **Regression (low)**: Multiplicative gating could interfere with gradient flow. But VE gating (same mechanism) didn't cause gradient issues.
- **Crash (none)**: Standard tensor ops, all shapes static, torch.compile compatible.
- Expected range: -0.003 to +0.005

## If it works

- Try wider gate: Linear(n_embd, n_embd) for per-dimension gating (more expressive but heavier)
- Try gating MLP output as well (attention + MLP both gated)
- Combine with other untested ideas (intermediate RoPE range)

## If it fails

- Output gating is dead for transformers at this scale (unlike RWKV/Mamba where it's critical for the recurrence)
- Try intermediate RoPE range (1500, 8000, 60000, 300000) — explicitly listed as untested
- FP8 retry with simpler approach (MLP-only, torch.float8_e4m3fn without custom autograd)
- DeltaNet/GLA remains the only untried high-potential architectural direction


## Résultats
- val_bpb: 1.067803
- peak_vram_mb: 12499.5
- status: discarded
