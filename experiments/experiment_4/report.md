GQA (Grouped Query Attention) with n_kv_head=2 to reduce parameters by ~14% and increase training throughput.

## What changed
- Added `N_KV_HEAD = 2` hyperparameter (was implicitly equal to n_head=4)
- `build_model_config` now uses `n_kv_head=N_KV_HEAD` instead of `n_kv_head=num_heads`
- This gives a 2:1 GQA ratio: 4 query heads share 2 KV heads
- No other changes to architecture, optimizer, or hyperparameters

## Parameter impact
- K projection per layer: 512x512 → 512x256 (save 131K)
- V projection per layer: 512x512 → 512x256 (save 131K)
- 8 layers: save 2.1M params
- VE embedding (last layer): 32768x512 → 32768x256 (save 8.4M)
- Total savings: ~10.5M params (~14% of model)

## Why
- GQA is the standard approach in modern LLMs (Llama 2/3, Mistral, Gemma) — proven to maintain quality while reducing compute
- 14% fewer parameters → faster forward/backward passes → more training steps in the fixed 5-minute budget
- More steps has been the dominant factor for convergence in this training regime (confirmed by the TOTAL_BATCH_SIZE=2^18 results)
- The SDPA code path already handles GQA via repeat_interleave (lines 100-103), so no kernel changes needed
- A 2:1 ratio is the mildest form of GQA, minimizing quality risk

## Expected outcome
- If speed gain dominates: improvement of ~0.002-0.005 in val_bpb from more training steps
- If KV capacity loss dominates: regression of ~0.003-0.010 from reduced key-value diversity
- Main risk is VE dimension halving (512→256) since VE is critical (exp1 showed +0.023 regression without VE)
- No crash/OOM risk — purely architectural reduction


## Résultats
- val_bpb: 1.088248
- peak_vram_mb: 11076.2
- status: discarded
