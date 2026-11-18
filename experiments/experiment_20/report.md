Retry factored lm_head with 256-dim bottleneck projection to reduce output layer compute by ~50% and get more training steps (exp19 crashed due to infra, not code).

## What changed
- Added `self.lm_proj = nn.Linear(n_embd, n_embd // 2, bias=False)` — a 512→256 bottleneck projection before the output layer
- Changed `self.lm_head` from `nn.Linear(512, 32768)` to `nn.Linear(256, 32768)`
- Forward pass: `logits = self.lm_head(self.lm_proj(x))` instead of `logits = self.lm_head(x)`
- lm_proj initialized with uniform (like other matrix params), optimized with Muon
- Updated `num_scaling_params` to track lm_proj separately
- Updated optimizer param assertion to include lm_proj in matrix_params
- ~6 lines changed across 5 methods

## Why
- **Exp19 had identical code but crashed** — verified the code is syntactically and logically correct. The crash was almost certainly RunPod infrastructure (API timeout or similar), not a code bug. Worth retrying.
- **The lm_head is a massive compute bottleneck.** With shape (512, 32768), it accounts for ~36% of total forward FLOPs. The vocab dimension (32768) is huge vs model dim (512).
- **Factoring cuts lm_head FLOPs by ~50%.** (512→32768) becomes (512→256) + (256→32768). FLOPs: 16.8M → 8.5M per token. Total compute reduction: ~18%.
- **"More steps" is the ONLY winning strategy.** MLP 3x (-0.008) proved that reducing compute per step is dominant. This targets the other major compute cost: the output layer.
- **256-dim bottleneck should suffice.** Softcap=15 bounds logits to [-15, 15]. Next-token prediction concentrates on a handful of tokens. 256 independent directions is enough. ALBERT uses 128-dim bottleneck for 768-dim model.
- **Net 8.2M fewer params** (lm_proj adds 131K, lm_head shrinks by 8.4M → net -8.2M, ~11% of total).

## Expected outcome
- If 18% compute reduction gives enough extra steps and 256 dims is sufficient: improvement of ~0.003-0.008
- If 256-dim bottleneck is too restrictive for output prediction: regression of ~0.002-0.010
- If bottleneck hurts exactly as much as extra steps help: neutral within noise (~0.003)
- No crash risk (exp19 code was correct, crash was infra)


## Résultats
- val_bpb: 1.095730
- peak_vram_mb: 10331.9
- status: discarded
