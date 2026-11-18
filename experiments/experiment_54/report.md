Multi-Query Attention (MQA, n_kv_head=1) to reduce KV parameters and VE embedding size for faster training throughput.

## What changed
- Added `N_KV_HEAD = 1` hyperparameter.
- Changed `build_model_config` to use `n_kv_head=N_KV_HEAD` instead of `n_kv_head=num_heads`.

## Why
- **GQA (n_kv_head=2) was essentially tied with baseline** (exp4: 1.088248 vs 1.088007, +0.000241). This shows the model doesn't need full MHA (4 KV heads). Going to MQA (1 KV head) saves even more compute.
- **~21% parameter reduction.** K/V projections shrink from (512,512) to (128,512) per layer. VE embedding shrinks from 32768x512 to 32768x128. Total savings: ~15.7M params.
- **Faster Muon optimizer step.** KV matrices are now in a smaller shape group (16 params of shape 128x512 vs 32 params of shape 512x512). Newton-Schulz on the smaller stacked tensor is faster.
- **More training steps in 5 minutes.** The dominant factor for this model's convergence is step count. Any compute savings translate to more steps.
- **Explicitly listed as untested promising direction** in prompt.md notes: "MQA (n_kv_head=1) -- GQA etait un tie, MQA pourrait aller plus loin."
- **The SDPA path already supports MQA.** The code has `repeat_interleave` for expanding KV heads to match query heads.

## Expected outcome
- If MQA maintains quality while gaining throughput: improvement of ~0.0005-0.001
- If single KV head loses too much representational capacity: regression of ~0.001-0.003
- If throughput gain is negligible: neutral within noise
- No crash, OOM, or compilation risk -- trivial config change, all code paths already support n_kv_head < n_head


## Résultats
- val_bpb: 1.085454
- peak_vram_mb: 9830.4
- status: discarded
