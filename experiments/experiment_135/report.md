Intermediate multi-scale RoPE range (1500, 8000, 60000, 300000) — 200x span vs current 64x span.

## What changed

One constant change in `train.py` (line 512):
- `ROPE_BASES = (2500, 10000, 40000, 160000)` -> `ROPE_BASES = (1500, 8000, 60000, 300000)`

Per-head base frequencies shifted to wider range:
- Head 0: 2500 -> 1500 (more local, resolves ~1-15 token distances)
- Head 1: 10000 -> 8000 (slightly more local)
- Head 2: 40000 -> 60000 (more global)
- Head 3: 160000 -> 300000 (very global, ~300-2000 token distances)

Overall span: 64x -> 200x. Zero new parameters, zero compute overhead.

## Why

- **Explicitly requested**: prompt.md piste #4 says "try intermediate (1500, 8000, 60000, 300000) before declaring dead." The wide range (exp109: 1000/5000/50000/500000, 500x) was neutral (+0.000025 vs then-baseline). The intermediate 200x range is the gap between current 64x and tested 500x.
- **Closes RoPE tuning thread**: If neutral, we declare multi-scale RoPE range tuning dead (the current 64x span from exp80 is the optimum). If it works, there's a new sweet spot to explore around.
- **Greater head specialization**: Head 0 at base=1500 can resolve finer-grained local patterns (bigrams, trigrams). Head 3 at base=300000 provides near-global context. The model can split its 4 heads across a wider range of distance scales.
- **All other architectural ideas exhausted**: MLP asymmetry (exp130), attention output gating (exp131), affine norm (exp132), post-emb norm removal (exp133), embedding LR (exp134) — all within noise of best.

## Risks

- **Neutral (high)**: Multi-scale RoPE (exp80) was only -0.0002 gain. The base range may not matter much beyond having some diversity. Both 64x and 500x produced similar results.
- **Regression (low)**: base=1500 might be too low (aliasing effects at short distances), base=300000 might be too high (provides ~constant positional info at T=2048, wasting a head).
- **Crash (none)**: Pure constant change, all RoPE computation is standard.
- Expected range: -0.002 to +0.003

## If it works

- Try even wider range: (1000, 10000, 100000, 500000) — 500x but with different distribution.
- Try asymmetric head count: 2 local heads + 1 medium + 1 global instead of 1 per scale.
- RoPE base tuning is a fertile direction worth further exploration.

## If it fails

- Multi-scale RoPE range is fully explored. Current (2500, 10000, 40000, 160000) is the optimum. Declare piste #4 dead.
- Remaining high-potential ideas: FP8 matmuls (retry with simpler approach), DeltaNet/GLA (complex but untested).
- Consider WARMDOWN_RATIO tuning (0.6 was never tested as a variable — exp13 showed 0.4 is much worse, suggesting more warmdown is better).


## Résultats
- val_bpb: 1.067172
- peak_vram_mb: 11984.2
- status: discarded
