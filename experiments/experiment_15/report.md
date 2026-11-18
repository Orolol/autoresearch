Reduce MLP hidden dimension from 3x to 2.5x to further decrease compute per step and get more training steps in 5 minutes.

## What changed
- `3 * config.n_embd` → `int(2.5 * config.n_embd)` in MLP c_fc and c_proj (hidden_dim: 1536 → 1280)
- Introduced explicit `hidden` variable for clarity
- 2 lines changed, no other modifications

## Why
- **Exp14 (4x→3x) was the biggest improvement ever**: 1.079602 (-0.008 from baseline). The "more steps" strategy is the only thing that has consistently beaten baseline.
- **Continuing the winning strategy**: If 4x→3x gave -0.008, then 3x→2.5x could give another meaningful improvement. The relationship between compute reduction and step count gain is roughly linear in this regime.
- **Capacity should be sufficient**: 1280 hidden units with ReluSquared still provides ample nonlinear capacity for a 512-dim model. Modern architectures (LLaMA SwiGLU) use ~2.67x effective expansion. 2.5x with ReluSquared is in the same ballpark.
- **~2.1M fewer params**: Saves ~3% of total parameters, meaning slightly less memory traffic and faster fwd/bwd per step.
- **No quality mechanism removed**: Same ReluSquared activation, same architecture. Just a narrower MLP bottleneck.

## Expected outcome
- If reduced compute gives enough extra steps to offset capacity loss: improvement of ~0.001-0.004
- If capacity loss exactly matches step gain: neutral within noise (~0.003)
- If 2.5x is too narrow and representational power degrades: regression of ~0.002-0.005
- No crash, OOM, or instability risk — pure dimension change


## Résultats
- val_bpb: 1.082175
- peak_vram_mb: 9795.4
- status: discarded
