Cosine warmdown schedule instead of linear warmdown to improve LR decay shape during the 60% warmdown phase.

## What changed
- Warmdown schedule: linear interpolation replaced by cosine interpolation
- `import math` added
- `get_lr_multiplier` warmdown branch: `cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC` replaced by `FINAL_LR_FRAC + 0.5 * (1.0 - FINAL_LR_FRAC) * (1 + math.cos(math.pi * warmdown_progress))`
- 2 lines changed total (1 import + 2 lines in function)

## Why
- **Cosine warmdown is the standard in modern LLM training.** WSD (Warmup-Stable-Decay) with cosine decay is used in most production LLMs (LLaMA, Chinchilla, PaLM). Our setup is WSD with linear decay — a less common variant.
- **Spends more time at higher LR.** Cosine warmdown decays slowly at first (staying near full LR longer) then drops faster at the end. This means more effective learning steps during the early warmdown phase, and a smoother transition to the minimum at the end.
- **Same average LR, different shape.** Both linear and cosine integrate to the same average over the warmdown window. The difference is entirely in dynamics — cosine gives better exploration early and better convergence late.
- **WARMDOWN_RATIO=0.6 means 60% of training is warmdown.** The shape of this curve matters significantly when most training is in the decay phase.
- **Zero compute overhead.** Single `math.cos` call per step (scalar, not tensor).
- **Never tested.** All 34 previous experiments used linear warmdown. This is the first schedule shape experiment.

## LR curve comparison (FINAL_LR_FRAC=0.0)
At warmdown start (progress=0.4): both = 1.0
At warmdown midpoint (progress=0.7): linear = 0.5, cosine = 0.854 (+71% higher)
At 75% warmdown (progress=0.85): linear = 0.25, cosine = 0.5 (+100% higher)
At warmdown end (progress=1.0): both = 0.0

The cosine curve keeps LR significantly higher through most of the warmdown, only catching up to linear near the very end.

## Expected outcome
- If cosine shape helps convergence: improvement of ~0.001-0.003
- If linear was already fine: neutral within noise (~0.003)
- If cosine keeps LR too high for too long: mild regression (~0.001-0.002) from insufficient cooling
- No crash, OOM, speed, or memory risk — pure schedule shape change


## Résultats
- val_bpb: 1.084106
- peak_vram_mb: 10337.7
- status: discarded
