Factored lm_head with 256-dim bottleneck projection to reduce output layer compute by ~50% and get more training steps.

## What changed
- Added `self.lm_proj = nn.Linear(512, 256, bias=False)` — a bottleneck projection before the output layer
- Changed `self.lm_head` from `nn.Linear(512, 32768)` to `nn.Linear(256, 32768)`
- Forward pass: `logits = self.lm_head(self.lm_proj(x))` instead of `logits = self.lm_head(x)`
- lm_proj initialized with uniform (like other matrix params), optimized with Muon
- ~6 lines changed

## Why
- **The lm_head is a massive compute bottleneck.** With shape (512, 32768), it accounts for ~36% of total forward FLOPs (1.1T per micro-step vs 1.9T for all 8 transformer layers combined). This is because the vocab dimension (32768) is huge compared to the model dimension (512).
- **Factoring the projection cuts lm_head FLOPs by ~50%.** Instead of one (512→32768) matmul, we do (512→256) + (256→32768). FLOPs: 1099G → 558G per micro-step. Total compute reduction: ~18%.
- **"More steps" is the ONLY winning strategy.** MLP 3x (-0.008) proved that reducing compute per step to get more optimizer steps is the dominant factor. This attack the same axis but on the output layer.
- **256-dim bottleneck should be sufficient.** With softcap=15 bounding logits to [-15, 15], the effective output distribution is constrained. Next-token prediction concentrates mass on a handful of tokens. 256 independent directions can express this well. Similar factored approaches work in ALBERT (128-dim bottleneck for 768-dim model).
- **lm_proj adds only 131K params** (0.2% of total) while removing 8.3M params from lm_head (11% of total). Net parameter reduction: ~8.2M.

## Expected outcome
- If 18% compute reduction gives enough extra steps and 256 dims is sufficient: improvement of ~0.003-0.008
- If 256-dim bottleneck is too restrictive for output prediction: regression of ~0.002-0.010
- If bottleneck hurts exactly as much as extra steps help: neutral within noise (~0.003)
- No crash, OOM, or instability risk — standard linear projection
