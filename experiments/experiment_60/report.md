Parallel attention + MLP (PaLM/GPT-J style) — both sublayers share the same normalized input instead of sequential processing.

## What changed

In Block.forward, replaced the sequential sublayer pattern:
```python
x = x + self.attn(norm(x), ve, cos_sin, window_size)
x = x + self.mlp(norm(x))
```
with a parallel pattern:
```python
nx = norm(x)
x = x + self.attn(nx, ve, cos_sin, window_size) + self.mlp(nx)
```

## Why

This is a genuine architectural change used in PaLM and GPT-J. The key differences:
1. **One fewer norm per block** (8 norms saved total) — small compute savings (~1-2%)
2. **MLP no longer sees attention output** — both sublayers operate on the same pre-norm input independently
3. **Cleaner gradient flow** — both sublayer gradients pass through the same normalization path

At this small model size (512 dim, 8 layers), the sequential information flow (MLP processing post-attention features) may not provide meaningful benefit since:
- Early/middle layer attention patterns are relatively noisy
- The residual stream already carries the original signal
- PaLM validated parallel works at larger scales where sequential flow would matter MORE

Important context: on RTX 5090 (not Hopper SM 9.0), SDPA is used with full causal attention — window patterns have zero effect on compute. Most compute-reduction ideas are exhausted. This experiment targets quality-per-step via architectural change.

## Risks

- No crash/OOM/speed risk (same params, same shapes, pure control flow change)
- If parallel is worse at this scale: mild regression ~0.001-0.005
- If the MLP benefits from seeing attention features in the sequential path, regression could be up to ~0.005


## Résultats
- val_bpb: 1.112265
- peak_vram_mb: 9311.2
- status: discarded
