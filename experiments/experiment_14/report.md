Reduce MLP hidden dimension from 4x to 3x to decrease compute per step and get more training steps in 5 minutes.

## What changed
- `nn.Linear(config.n_embd, 4 * config.n_embd)` → `nn.Linear(config.n_embd, 3 * config.n_embd)` in MLP.c_fc
- `nn.Linear(4 * config.n_embd, config.n_embd)` → `nn.Linear(3 * config.n_embd, config.n_embd)` in MLP.c_proj
- 2 lines changed, no other modifications

## Why
- **"More steps" is the ONLY strategy that has ever beaten baseline.** Every architectural addition that increased compute per step (MTP, parallel attn+MLP) made things worse. Every removal that reduced quality (VE, x0_lambdas, softcap) also made things worse. The sweet spot is: same quality, less compute.
- **MLP is the largest compute consumer in the transformer.** With 4x expansion, MLP accounts for ~44% of transformer matrix params (2 × 512 × 2048 × 8 layers = 16.8M out of ~25M transformer matrix params). Reducing to 3x saves ~4.2M params (5.5% of total model).
- **The model is severely undertrained.** With only ~830 steps in 5 minutes, the 4x MLP capacity may not be fully utilized. Modern architectures (LLaMA with SwiGLU) use ~2.67x effective expansion. Even 3x with ReluSquared should provide sufficient nonlinear capacity.
- **No quality mechanism removed.** Unlike VE removal (+0.023 regression) or x0_lambdas removal (+0.020 regression), reducing MLP width doesn't remove any architectural mechanism. It just makes an existing mechanism slightly smaller.
- **Faster forward/backward → more optimizer steps → better convergence** in the time-limited training regime.

## Expected outcome
- If reduced compute gives enough extra steps to offset capacity loss: improvement of ~0.001-0.003
- If capacity loss exactly matches step gain: neutral within noise (~0.003)
- If 3x is too small and the model can't learn: regression of ~0.002-0.005
- No crash, OOM, or instability risk — pure dimension change


## Résultats
- val_bpb: 1.079602
- peak_vram_mb: 10337.7
- status: IMPROVED
