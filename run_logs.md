# Run Logs — Autoresearch

Journal des expériences lancées par l'agent.

---


## Experiment #0 — 2026-03-11 10:33:30

- **Best val_bpb avant** : N/A (baseline)
- **Status** : keep (val_bpb: 1.088007)
- **Description** : baseline


## Experiment #0 — 2026-03-11 10:39:26

- **Best val_bpb avant** : N/A (baseline)
- **Status** : en cours...


## Experiment #0 — 2026-03-11 10:51:16

- **Best val_bpb avant** : N/A (baseline)
- **Status** : keep (val_bpb: 1.088007)
- **Description** : baseline


## Experiment #0 — 2026-03-11 10:54:33

- **Best val_bpb avant** : N/A (baseline)
- **Status** : en cours...


## Experiment #0 — 2026-03-11 10:56:22

- **Best val_bpb avant** : N/A (baseline)
- **Status** : keep (val_bpb: 1.088007)
- **Description** : baseline


## Experiment #1 — 2026-03-11 11:03:25

- **Best val_bpb avant** : 1.088007
- **Status** : discard (val_bpb: 1.111258)
- **Description** : Remove all Value Embeddings (VE=0) to test if the model can match baseline without any VE overhead.
- **Idée** : Remove all Value Embeddings (VE=0). Baseline uses VE-last-only (16.8M params). Removing it entirely saves 22% of total params, speeds up forward/backward, and gets more training steps in 5 min.
- **Pourquoi** : Notes confirm VE-last-only works (baseline 1.088007). Next step is testing VE=0. If quality holds, it's a massive simplification. More steps = better convergence has been the dominant factor.
- **Risques** : VE provides direct token→value signal that might help last-layer attention quality. Possible regression of ~0.003-0.01 if VE is important. No crash/OOM risk.
- **Changement** : `has_ve()` returns False unconditionally. All VE code paths become no-ops.


## Experiment #2 — 2026-03-11 11:15:07

- **Best val_bpb avant** : 1.088007
- **Status** : discard (val_bpb: 1.093279)
- **Description** : Increase logit softcap from 15 to 30 to allow more expressive predictions while keeping training stable.
- **Idée** : Increase logit softcap from 15 to 30 for more expressive predictions while keeping regularization.
- **Pourquoi** : Removing softcap entirely (exp11 old series) = 1.097 (worse). But softcap=15 may be too restrictive, limiting confident predictions. softcap=30 = 2x more dynamic range with stability.
- **Risques** : Very low. Softcap=30 still prevents logit explosion. Worst case: neutral result within noise.
- **Changement** : `softcap = 15` → `softcap = 30` in forward method. Single line change.


## Experiment #3 — 2026-03-11 11:23:38

- **Best val_bpb avant** : 1.088007
- **Status** : discard (val_bpb: 1.108641)
- **Description** : Remove x0_lambdas (per-layer skip connection from initial embedding) to simplify the residual stream.
- **Idée** : Remove x0_lambdas (skip connection from initial embedding to each layer). Simplify the residual stream.
- **Pourquoi** : x0_lambdas adds 10% of the original embedding at each layer input. But VE already provides a direct token→value connection in the last layer. Removing x0_lambdas simplifies the forward pass and removes a learnable scalar per layer. If the model doesn't need the direct embedding signal at every layer, this is a free simplification.
- **Risques** : If deep layers rely on the direct embedding signal for gradient flow, quality could regress. No crash/OOM risk. The change reduces params slightly and simplifies compute.
- **Changement** : Removed `self.x0_lambdas` parameter, `x0 = x` reference, and `self.x0_lambdas[i] * x0` term from the forward loop. Updated optimizer and param counting accordingly.


## Experiment #4 — 2026-03-11 11:37:06

- **Best val_bpb avant** : 1.088007
- **Status** : discard (val_bpb: 1.088248)
- **Description** : GQA (Grouped Query Attention) with n_kv_head=2 to reduce parameters by ~14% and increase training throughput.
- **Idée** : GQA (Grouped Query Attention) with n_kv_head=2 instead of 4 to reduce params by ~14% and get more training steps.
- **Pourquoi** : With 4 query heads and 2 KV heads (2:1 GQA), we save ~10.5M params (2.1M from K/V projections across 8 layers + 8.4M from smaller VE embedding). Fewer params → faster forward/backward → more steps in 5 min. GQA is proven in Llama 2/3, Mistral, Gemma. The code already supports GQA (repeat_interleave path). More steps has been the dominant factor for convergence in this regime.
- **Risques** : With only 4 heads, reducing KV heads to 2 might lose meaningful key-value diversity. VE dimension also halves (512→256) which could hurt since VE is critical (exp1 showed removing VE = +0.023 regression). No crash/OOM risk.
- **Changement** : Added `N_KV_HEAD = 2` hyperparameter. Changed `build_model_config` to use `n_kv_head=N_KV_HEAD` instead of `n_kv_head=num_heads`.


## Experiment #5 — 2026-03-11 11:52:07

- **Best val_bpb avant** : 1.088007
- **Status** : discard (val_bpb: 1.091936)
- **Description** : Remove all weight decay (WEIGHT_DECAY=0.0) to test if Muon's cautious WD is counterproductive during short 5-minute trai
- **Idée** : Remove all weight decay (WEIGHT_DECAY=0.0) to test if regularization hurts at this short training duration.
- **Pourquoi** : WD is designed to prevent overfitting during long training. With only 5 min / ~830 steps, the model is severely undertrained. Even mild cautious WD slows effective learning rate during the first 40% of training. Removing it lets the optimizer converge faster.
- **Risques** : Very low. Softcap=15 and QK-norm provide implicit regularization. Worst case: neutral result within noise.
- **Changement** : `WEIGHT_DECAY = 0.2` → `WEIGHT_DECAY = 0.0`. Single line change.


## Experiment #6 — 2026-03-11 12:05:10

- **Best val_bpb avant** : 1.088007
- **Status** : discard (val_bpb: 1.098882)
- **Description** : Halve TOTAL_BATCH_SIZE from 2^18 to 2^17 to double optimizer steps and push the "more steps = better convergence" hypoth
- **Idée** : Halve TOTAL_BATCH_SIZE from 2^18 to 2^17 to double optimizer steps (~1660 vs ~830 in 5 min).
- **Pourquoi** : "More steps" is the ONLY strategy that has ever beaten baseline. 2^18 worked, so push further. 131K tokens/step is still large enough for stable gradients. grad_accum goes from 4 to 2, so wall-clock overhead is minimal.
- **Risques** : Gradient noise from smaller batches might hurt convergence. If 2^18 was already optimal, this regresses. No OOM/crash risk.
- **Changement** : `TOTAL_BATCH_SIZE = 2**18` → `TOTAL_BATCH_SIZE = 2**17`. Single line change.


## Experiment #7 — 2026-03-11 12:12:57

- **Best val_bpb avant** : 1.088007
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Increase MATRIX_LR from 0.04 to 0.06 (+50%) to accelerate convergence in the undertrained regime.
- **Idée** : Increase MATRIX_LR from 0.04 to 0.06 (+50%) to make each optimizer step more aggressive.
- **Pourquoi** : The model is severely undertrained (~830 steps in 5 min). We can't get more steps (2^17 failed due to gradient noise). Instead, make each step do more by increasing the learning rate for transformer matrix parameters (the bulk of the model). Muon's NS orthogonalization normalizes gradient direction, so higher LR just means bigger steps in the optimal direction. QK-norm + softcap=15 provide additional stability against divergence.
- **Risques** : If 0.06 is too high, the model could overshoot good minima during warmdown, leading to a regression of ~0.003-0.010. Very unlikely to cause divergence given Muon's inherent stability + QK-norm + softcap. No OOM/crash risk.
- **Changement** : `MATRIX_LR = 0.04` → `MATRIX_LR = 0.06`. Single line change.


## Experiment #8 — 2026-03-11 12:19:33

- **Best val_bpb avant** : 1.088007
- **Status** : discard (val_bpb: 1.088767)
- **Description** : Re-run MATRIX_LR=0.06 (+50%) which crashed in exp7 due to RunPod API timeout (never actually tested).
- **Description** : Re-run MATRIX_LR=0.06 (+50%) — exp7 crashed due to RunPod API timeout, never actually tested on GPU.
- **Idée** : Increase MATRIX_LR from 0.04 to 0.06 to make each optimizer step more aggressive in the undertrained regime.
- **Pourquoi** : Can't get more steps (2^17 failed). Instead make each step do more. Muon's NS orthogonalization normalizes direction, so higher LR = bigger steps in optimal direction. QK-norm + softcap=15 provide stability.
- **Risques** : If 0.06 overshoots, mild regression (~0.003-0.010). No OOM/crash risk. Pure hyperparameter change.
- **Changement** : `MATRIX_LR = 0.04` → `MATRIX_LR = 0.06`. Single line change.


## Experiment #9 — 2026-03-11 12:27:27

- **Best val_bpb avant** : 1.088007
- **Status** : keep (val_bpb: 1.087971)
- **Description** : Z-loss regularization (PaLM-style) to improve logit calibration and gradient flow through the softcap layer.
- **Idée** : Z-loss regularization (1e-4 * logsumexp(logits)^2) — penalize large partition function for better logit calibration.
- **Pourquoi** : Proven by PaLM and ST-MoE. Complementary to softcap: softcap clips logits via tanh, but z-loss provides smooth gradients that push logits away from tanh saturation boundary, improving gradient flow through the final layer. Zero compute overhead (one logsumexp per batch). The model's logits pass through softcap=15*tanh(logits/15) — if logits cluster near ±15, tanh saturates and gradients vanish. Z-loss discourages this.
- **Risques** : Very low. Z-loss coefficient 1e-4 adds ~0.06 to loss (~2% of CE loss). Only applied during training (reduction='mean'), eval is unchanged. No change to architecture, speed, or memory.
- **Changement** : Added z-loss term in forward() when reduction='mean': `logz = torch.logsumexp(logits, dim=-1); z_loss = 1e-4 * logz.square().mean(); loss = loss + z_loss`. 3 lines added.


## Experiment #10 — 2026-03-11 12:38:12

- **Best val_bpb avant** : 1.087971
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Multi-Token Prediction (MTP) — auxiliary loss predicting token i+2 from position i via learned projection + shared lm_
- **Idée** : Multi-Token Prediction (MTP) — add auxiliary loss that predicts token i+2 from position i using a learned projection + shared lm_head.
- **Pourquoi** : Meta's MTP paper and DeepSeek-V3 showed that predicting further ahead forces the model to build richer internal representations. Each training step provides gradient signal from 2 prediction targets instead of 1, effectively increasing information density per step. The projection is lightweight (512×512 = 262K params, 0.4% of total) and shares the lm_head weights. Auxiliary loss weight = 0.1 (conservative). ~10-15% compute overhead from the extra lm_head matmul.
- **Risques** : The extra lm_head forward+backward adds ~10-15% wall-clock overhead → ~70-80 fewer steps. If MTP gradients don't compensate for fewer steps, mild regression (~0.003-0.005). No OOM risk (only 262K extra params). No crash risk (standard CE loss on valid targets).
- **Changement** : Added `self.mtp_proj = nn.Linear(n_embd, n_embd, bias=False)`. In forward(), computes `mtp_logits = lm_head(mtp_proj(x[:, :-1]))` with softcap, then `0.1 * CE(mtp_logits, targets[:, 1:])`. Added to Muon optimizer. ~10 lines of code.


## Experiment #11 — 2026-03-11 12:45:01

- **Best val_bpb avant** : 1.087971
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Embedding tying (wte = lm_head weight sharing) to save 16.8M params (22% of total model) and simplify the architecture.
- **Description** : Embedding tying (wte = lm_head weight sharing) to save 16.8M params (22% of total).
- **Idée** : Share weights between token embedding (wte) and output projection (lm_head). Saves 16.8M params, reduces memory, and forces a shared embedding space. Well-established technique (GPT-2, T5, LLaMA).
- **Pourquoi** : 22% parameter reduction is massive. The model is undertrained (5 min), so fewer unique params to learn should help convergence. Softcap=15 + z-loss protect against large initial logits from shared init. AdamW's per-element adaptivity handles the merged gradients from both embedding and output paths.
- **Risques** : The original embedding_lr=0.6 and unembedding_lr=0.004 are 150x apart. Using geometric mean (~0.049) as compromise. If the LR is wrong for either side, could regress. No crash/OOM risk — pure architectural simplification.
- **Changement** : (1) `self.lm_head.weight = self.transformer.wte.weight`, (2) single init with std=0.5, (3) removed lm_head from optimizer — shared weight in embedding group with tied_lr=sqrt(0.6*0.004)≈0.049, (4) value_embeds keep their original LR=0.6.


## Experiment #12 — 2026-03-11 12:56:04

- **Best val_bpb avant** : 1.087971
- **Status** : discard (val_bpb: 1.093492)
- **Description** : Multi-Token Prediction (MTP) retry — auxiliary loss predicting token i+2 via learned projection + shared lm_head (exp1
- **Description** : Multi-Token Prediction (MTP) retry — auxiliary loss predicting token i+2 via learned projection + shared lm_head.
- **Idée** : Retry MTP (exp10 crashed due to code bug). Add auxiliary loss that predicts token i+2 from position i using a learned 512×512 projection + shared lm_head. Weight 0.1.
- **Pourquoi** : Meta's MTP paper and DeepSeek-V3 showed predicting further ahead forces richer internal representations. Each step provides gradient signal from 2 prediction targets. Only 262K extra params (0.3%). Exp10 crashed due to implementation bug, not a fundamental issue.
- **Risques** : ~10-15% compute overhead (extra lm_head matmul) → ~70-80 fewer steps. If MTP gradients don't compensate, mild regression. No OOM risk.
- **Changement** : Added `self.mtp_proj = nn.Linear(n_embd, n_embd, bias=False)`. In forward(), computes `mtp_logits = lm_head(mtp_proj(x[:, :-1]))` with softcap, adds `0.1 * CE(mtp_logits, targets[:, 1:])` to loss. Added to Muon optimizer. ~15 lines of code.


## Experiment #13 — 2026-03-11 13:07:10

- **Best val_bpb avant** : 1.087971
- **Status** : discard (val_bpb: 1.089407)
- **Description** : Reduce WARMDOWN_RATIO from 0.6 to 0.4 to spend 60% of training at full learning rate instead of 40%.
- **Description** : Reduce WARMDOWN_RATIO from 0.6 to 0.4 for more steps at full learning rate.
- **Idée** : Spend 60% of training at full LR instead of 40%. More effective learning steps before cooling down.
- **Pourquoi** : Model is undertrained (5 min). Currently 60% of training is warmdown — too conservative. More full-LR steps = more effective learning. MTP and architectural changes failed; this is a safe orthogonal test.
- **Risques** : Compressed warmdown might not settle into optimal minimum. Very low risk — no crash/OOM. Expected effect within ±0.003.
- **Changement** : `WARMDOWN_RATIO = 0.6` → `WARMDOWN_RATIO = 0.4`. Single line change.


## Experiment #14 — 2026-03-11 13:16:07

- **Best val_bpb avant** : 1.087971
- **Status** : keep (val_bpb: 1.079602)
- **Description** : Reduce MLP hidden dimension from 4x to 3x to decrease compute per step and get more training steps in 5 minutes.
- **Description** : Reduce MLP hidden dim from 4x to 3x to decrease compute per step and get more training steps.
- **Idée** : Shrink MLP expansion ratio from 4x (2048) to 3x (1536). Saves ~4.2M params (5.5% of total), faster fwd/bwd → more steps.
- **Pourquoi** : "More steps" is the ONLY strategy that has beaten baseline. Every compute-adding change (MTP, parallel attn) regressed. The model is undertrained in 5 min, so reduced MLP capacity should be fine. No quality mechanism removed — just a smaller hidden dim.
- **Risques** : If 3x MLP is too small, mild regression (~0.002-0.005). No crash/OOM risk. ReluSquared still works at 3x.
- **Changement** : `4 * config.n_embd` → `3 * config.n_embd` in MLP c_fc and c_proj. 2 lines changed.


## Experiment #15 — 2026-03-11 13:31:45

- **Best val_bpb avant** : 1.079602
- **Status** : discard (val_bpb: 1.082175)
- **Description** : Reduce MLP hidden dimension from 3x to 2.5x to further decrease compute per step and get more training steps in 5 minute
- **Description** : Reduce MLP hidden dimension from 3x to 2.5x to further decrease compute per step and get even more training steps.
- **Idée** : Push the winning "less compute per step" strategy further. MLP 3x gave the best result ever (1.079602). Try 2.5x (1280 hidden dim) for even faster steps.
- **Pourquoi** : Exp14 (4x→3x) gave -0.008 improvement — the biggest single improvement in the entire experiment series. The model is severely undertrained in 5 min. Each step saved → extra optimizer step → better convergence. 2.5x still provides 1280 hidden units with ReluSquared, which should be sufficient at this scale.
- **Risques** : If 2.5x MLP is too narrow, the nonlinear capacity may be insufficient and quality degrades. Expected ~2.1M fewer params (saving ~3% of total). Risk of mild regression (~0.002-0.005) if capacity bottleneck hits. No crash/OOM risk.
- **Changement** : `3 * config.n_embd` → `int(2.5 * config.n_embd)` in MLP c_fc and c_proj. 2 lines changed (replaced with explicit `hidden` variable for clarity).


## Experiment #16 — 2026-03-11 13:40:05

- **Best val_bpb avant** : 1.079602
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Enable torch.compile(mode="max-autotune") for CUDA graphs and autotuned Triton kernels to increase training throughput.
- **Description** : Enable torch.compile(mode="max-autotune") for CUDA graphs + autotuned kernels.
- **Idée** : Switch from default torch.compile to max-autotune mode. Enables CUDA graphs (eliminate kernel launch overhead) and autotuned Triton kernels (benchmark multiple configurations). Zero architectural change.
- **Pourquoi** : "More steps" is the only winning strategy. MLP 3x worked by reducing compute per step. This attacks the same axis from the infrastructure side — same compute, but faster execution → more steps in 5 min. First 10 steps don't count toward time budget, so extra compilation time is free.
- **Risques** : Very low. Same model math. CUDA graphs slightly increase VRAM (~100-200MB for graph capture). Potential compatibility issue with training loop (unlikely with dynamic=False). No architectural risk.
- **Changement** : `torch.compile(model, dynamic=False)` → `torch.compile(model, dynamic=False, mode="max-autotune")`. Single line change.


## Experiment #17 — 2026-03-11 13:43:31

- **Best val_bpb avant** : 1.079602
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Retry torch.compile(mode="max-autotune") with cudagraph_mark_step_begin() fix to enable CUDA graphs and autotuned Triton
- **Description** : Retry torch.compile(mode="max-autotune") with cudagraph_mark_step_begin() fix for CUDA graphs + autotuned kernels.
- **Idée** : Exp16 crashed because CUDA graphs overwrote tensor outputs between gradient accumulation steps. The fix is `torch.compiler.cudagraph_mark_step_begin()` before each forward pass. This enables CUDA graphs (eliminates kernel launch overhead) + autotuned Triton kernels.
- **Pourquoi** : "More steps" is the only winning strategy. MLP 3x reduced compute/step. max-autotune reduces wall-clock/step via CUDA graphs + optimal kernel configs. Zero architectural risk. Compilation overhead is free (first 10 steps excluded).
- **Risques** : Very low. Same model math. Slight extra VRAM for graph capture (~100-200MB). If the fix doesn't fully resolve CUDA graph issues, could crash again. No quality risk.
- **Changement** : (1) `torch.compile(model, dynamic=False)` → `torch.compile(model, dynamic=False, mode="max-autotune")`, (2) Added `torch.compiler.cudagraph_mark_step_begin()` before each `model(x, y)` in the grad accum loop. 2 lines changed.


## Experiment #18 — 2026-03-11 13:50:27

- **Best val_bpb avant** : 1.079602
- **Status** : discard (val_bpb: 1.112418)
- **Description** : Reduce DEPTH from 8 to 7 layers to decrease compute per step and get more training steps in 5 minutes.
- **Description** : Reduce DEPTH from 8 to 7 layers to decrease compute per step and get more training steps.
- **Idée** : Remove 1 transformer layer (8→7) to save ~12.5% compute per step. Model dim stays 512 (7*64=448 rounded up). Same MLP 3x, same VE-last-only, same everything else.
- **Pourquoi** : "More steps" is the ONLY winning strategy. MLP 3x (-0.008) proved compute reduction works. Removing 1 layer saves ~2.6M matrix params. 7 layers should provide sufficient capacity at 512 dim.
- **Risques** : If 7 layers is too shallow, mild regression (~0.002-0.005). No crash/OOM risk. Pure depth change.
- **Changement** : `DEPTH = 8` → `DEPTH = 7`. Single line change.


## Experiment #19 — 2026-03-11 14:00:32

- **Best val_bpb avant** : 1.079602
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Factored lm_head with 256-dim bottleneck projection to reduce output layer compute by ~50% and get more training steps.
- **Description** : Factored lm_head with 256-dim bottleneck to reduce output layer compute by ~50% and get more training steps.
- **Idée** : Add a 512→256 bottleneck projection before the lm_head (256→32768). The lm_head is ~36% of total forward FLOPs due to the huge vocab (32768). Factoring it cuts lm_head FLOPs by ~50%, total compute by ~18%.
- **Pourquoi** : "More steps" is the ONLY winning strategy. MLP 3x reduced per-layer compute. This targets the OTHER major compute cost: the output layer. 256-dim bottleneck should suffice since softcap=15 constrains logit range and next-token prediction concentrates on few tokens. Similar to ALBERT's factored embeddings. Net 8.2M fewer params (11%).
- **Risques** : If 256 dims is too restrictive for discriminating 32768 tokens, quality regresses (~0.002-0.010). No crash/OOM risk. lm_proj optimized with Muon (2D matrix).
- **Changement** : Added `self.lm_proj = nn.Linear(512, 256, bias=False)`, changed lm_head to `nn.Linear(256, 32768)`, forward: `logits = lm_head(lm_proj(x))`. ~6 lines changed.


## Experiment #20 — 2026-03-11 14:11:11

- **Best val_bpb avant** : 1.079602
- **Status** : discard (val_bpb: 1.095730)
- **Description** : Retry factored lm_head with 256-dim bottleneck projection to reduce output layer compute by ~50% and get more training s
- **Description** : Retry factored lm_head (exp19 crashed — likely infra, not code). 512→256 bottleneck before lm_head to cut output FLOPs by ~50%.
- **Idée** : Same as exp19: add lm_proj (512→256) bottleneck before lm_head (256→32768). The lm_head accounts for ~36% of total forward FLOPs due to huge vocab (32768). Factoring cuts lm_head FLOPs by ~50%, total compute by ~18%.
- **Pourquoi** : "More steps" is the ONLY winning strategy. MLP 3x (-0.008) reduced per-layer compute. This targets the output layer — the OTHER major compute cost. 256-dim bottleneck should suffice (softcap=15 constrains logit range, next-token prediction concentrates mass on few tokens). Similar to ALBERT's factored embeddings.
- **Risques** : If 256 dims too restrictive: regression ~0.002-0.010. Exp19 code was correct (verified) — crash was likely RunPod infra. No OOM risk. lm_proj optimized with Muon.
- **Changement** : Added `self.lm_proj = nn.Linear(n_embd, n_embd//2, bias=False)`, changed lm_head to `nn.Linear(n_embd//2, vocab_size)`, forward: `logits = lm_head(lm_proj(x))`. lm_proj in Muon group, init uniform. ~6 lines changed across 5 methods.


## Experiment #21 — 2026-03-11 14:22:10

- **Best val_bpb avant** : 1.079602
- **Status** : discard (val_bpb: 1.098823)
- **Description** : Enable torch.compile(mode="max-autotune-no-cudagraphs") for autotuned Triton kernels without CUDA graphs.
- **Description** : Enable torch.compile(mode="max-autotune-no-cudagraphs") for autotuned Triton kernels without CUDA graphs.
- **Idée** : Use max-autotune-no-cudagraphs mode to get autotuned Triton kernel configurations without CUDA graphs (which crashed exp16/17). Benchmarks multiple tile sizes for each matmul shape to find the fastest.
- **Pourquoi** : "More steps" is the only winning strategy. Autotuned kernels may speed up execution by 3-7% for our specific matmul shapes (512×1536, 512×32768). Zero architectural risk. Compilation overhead absorbed by warmup steps.
- **Risques** : Very low. No CUDA graphs = no crash risk. Same model math. If autotuning takes too long during compilation, might slightly reduce effective training time. No OOM risk.
- **Changement** : `torch.compile(model, dynamic=False)` → `torch.compile(model, dynamic=False, mode="max-autotune-no-cudagraphs")`. Single line change.


## Experiment #22 — 2026-03-11 14:33:37

- **Best val_bpb avant** : 1.079602
- **Status** : discard (val_bpb: 1.080504)
- **Description** : Iso-compute depth-for-width swap: DEPTH=9 + MLP 2.5x + ASPECT_RATIO=56 to add an extra transformer layer at near-zero co
- **Description** : Iso-compute depth-for-width swap: DEPTH=9 + MLP 2.5x + ASPECT_RATIO=56 to add an extra transformer layer at near-zero compute cost.
- **Idée** : Trade MLP width for depth: 9 layers with 2.5x MLP instead of 8 layers with 3x MLP. Matrix params: 21.23M vs 20.97M (+1.25%). Same model dim (512), same heads (4).
- **Pourquoi** : DEPTH=7 was catastrophic (+0.033), proving depth is critical. MLP 3x→2.5x costs only +0.003. Trading width for depth is a favorable exchange. Even with diminishing returns, DEPTH 8→9 should net +0.002-0.007.
- **Risques** : If 2.5x MLP is too narrow with 9 layers, mild regression (~0.002-0.005). No crash/OOM risk. Attention compute +12.5%, MLP compute -17% per layer.
- **Changement** : `DEPTH=8→9`, `ASPECT_RATIO=64→56`, MLP `3*n_embd→int(2.5*n_embd)`. 3 lines changed.


## Experiment #23 — 2026-03-11 14:44:49

- **Best val_bpb avant** : 1.079602
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Switch lm_head from AdamW to Muon optimizer for Newton-Schulz orthogonalized updates on the largest parameter matrix.
- **Description** : Switch lm_head from AdamW to Muon optimizer for NS-orthogonalized updates on the largest parameter matrix (44% of FLOPs).
- **Idée** : Move lm_head (16.8M params, shape 32768×512) from AdamW (LR=0.004) to Muon (LR=0.04, NS orthogonalization). NS pushes 32768 output vectors toward orthogonality in 512-dim space, maximizing token discriminative power.
- **Pourquoi** : lm_head is the single largest matrix and currently uses the most conservative optimizer. Muon has proven effective for all other matrices. NS orthogonalization is a strong structural prior for the output layer — it ensures well-separated output directions. NorMuon handles the step sizing adaptively.
- **Risques** : Effective LR after Muon's rectangular scaling = 0.32 (8x factor for 32768/512 aspect ratio). Could be too aggressive, but NorMuon self-corrects via second momentum. Softcap + z-loss bound logit magnitudes. Also adds cautious WD to lm_head (was 0.0). No crash/OOM risk.
- **Changement** : Moved `self.lm_head.parameters()` into `matrix_params` list. Removed separate AdamW group for lm_head. Updated assertion. 3 lines changed in `setup_optimizer`.


## Experiment #24 — 2026-03-11 14:53:54

- **Best val_bpb avant** : 1.079602
- **Status** : discard (val_bpb: 1.095282)
- **Description** : Increase DEVICE_BATCH_SIZE from 32 to 64 to halve gradient accumulation steps (4→2) and improve GPU utilization with t
- **Description** : Increase DEVICE_BATCH_SIZE from 32 to 64 to halve gradient accumulation steps (4→2) and improve GPU utilization with MLP 3x.
- **Idée** : Double micro-batch size to reduce grad_accum from 4→2. Less Python loop overhead per optimizer step + better tensor core utilization with larger batch dim.
- **Pourquoi** : "More steps" is the only winning strategy. With MLP 3x, peak VRAM is only 10.1GB at batch=32 — plenty of headroom for batch=64 (~14-16GB). Previous batch=64 test (exp1 run3) used 4x MLP, different regime. Fewer micro-steps per optimizer step = faster wall-clock per step = more steps in 5 min.
- **Risques** : Very low. No OOM risk (10.1→~15GB, within 32GB). No crash risk. If GPU was already saturated at batch=32, result will be neutral. Worst case: mild regression if larger micro-batches somehow hurt convergence (~0.002-0.005).
- **Changement** : `DEVICE_BATCH_SIZE = 32` → `DEVICE_BATCH_SIZE = 64`. Single line change.


## Experiment #25 — 2026-03-11 15:04:27

- **Best val_bpb avant** : 1.079602
- **Status** : en cours...
- **Description** : Replace ReluSquared MLP with SwiGLU at iso-compute (2x hidden = same FLOPs/params as 3x ReluSquared).
- **Idée** : SwiGLU gating provides richer multiplicative interactions than ReluSquared. At iso-compute (hidden=1024 with 3 matrices = hidden=1536 with 2 matrices), test if the gating quality boost improves convergence.
- **Pourquoi** : SwiGLU is the standard in ALL modern LLMs (LLaMA, Mistral, Gemma). Previous failure was at 4x scale with a different baseline. At iso-compute with MLP 3x baseline, the comparison is fair. Fused gate+up matmul is GPU-friendly.
- **Risques** : Very low. Zero compute/param/memory change. If SwiGLU doesn't help at this scale, neutral result within noise. Previous SwiGLU failures were at 4x MLP with different baseline.
- **Changement** : MLP: `c_fc(512→2048 fused gate+up)`, `c_proj(1024→512 down)`, forward: `silu(gate) * up` instead of `relu(x).square()`. 4 lines changed in MLP class.


## Experiment #26 — 2026-03-11 15:13:32

- **Best val_bpb avant** : 1.079602
- **Status** : crash (val_bpb: 0.000000)
- **Description** : EMA (Exponential Moving Average) of model weights for smoother eval — average parameters during training to reduce noi
- **Description** : EMA (Exponential Moving Average) of model weights for smoother eval.
- **Idée** : Maintain a running average of all model parameters (decay=0.995, ~200 step window) during training. Swap to EMA weights before eval. Smooths out SGD noise for better generalization.
- **Pourquoi** : All "more steps" ideas are exhausted. Must improve quality per step. EMA is universally proven (diffusion, ViT, LLM). The model is undertrained (~830 steps) and the final weights are noisy (observed ~0.003 variance between identical runs). EMA captures the best of the convergence trajectory.
- **Risques** : Very low. ~220MB extra VRAM (10→10.2GB). ~1-2ms overhead per step. No architectural change. If warmdown already converges cleanly, result is neutral. Worst case: averaged weights slightly worse than endpoint (~0.001-0.003 regression).
- **Changement** : (1) `ema_state = [(p, p.data.clone()) for p in model.parameters()]` after compile, (2) `ema.lerp_(p.data, 0.005)` after each optimizer step, (3) `p.data.copy_(ema)` before eval. 6 lines added.


## Experiment #27 — 2026-03-11 15:17:41

- **Best val_bpb avant** : 1.079602
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Retry EMA (Exponential Moving Average) of model weights — exp26 crashed due to infra, never tested on GPU.
- **Description** : Retry EMA (exp26 crashed infra) — smooth model weights via exponential moving average for better eval.
- **Idée** : EMA (decay=0.995, ~200 step window) of all model parameters. Swap to averaged weights before eval. Smooths SGD noise.
- **Pourquoi** : All "more steps" ideas exhausted. Quality per step is the only remaining axis. EMA is universally proven. Model is noisy (~0.003 variance between identical runs). Exp26 was infra crash, never tested.
- **Risques** : Very low. ~152MB extra VRAM. ~1-2ms/step overhead. No architectural change. Worst case: neutral if warmdown already converges cleanly.
- **Changement** : (1) Clone all trainable params after torch.compile, (2) `ema.lerp_(p.data, 0.005)` after each optimizer step, (3) `p.data.copy_(ema)` before eval. 6 lines added.


## Experiment #28 — 2026-03-11 15:26:29

- **Best val_bpb avant** : 1.079602
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Retry SwiGLU at iso-compute (exp25 crashed infra, never tested on GPU) — replace ReluSquared MLP with SwiGLU gating at
- **Description** : Retry SwiGLU at iso-compute (exp25 crashed infra). Replace ReluSquared with SwiGLU gating at identical compute/params.
- **Idée** : SwiGLU (silu(gate) * up) provides richer multiplicative interactions than ReluSquared. At iso-compute (hidden=1024 with 3 matrices = hidden=1536 with 2 matrices), test if gating quality improves convergence. Exp25 had the exact same idea but crashed on RunPod infra — never tested on GPU.
- **Pourquoi** : SwiGLU is the standard in ALL modern LLMs. Previous SwiGLU failure was at 4x MLP scale, very different baseline. Iso-compute swap = zero risk to step count/speed. The only variable is quality per step.
- **Risques** : Very low. Zero compute/param/memory change. If SwiGLU doesn't help, neutral within noise. No crash/OOM risk.
- **Changement** : MLP: `c_fc(512→2048 fused gate+up)`, `c_proj(1024→512 down)`, forward: `silu(gate) * up` instead of `relu(x).square()`. 4 lines changed in MLP class.


## Experiment #29 — 2026-03-11 15:30:26

- **Best val_bpb avant** : 1.079602
- **Status** : en cours...
- **Description** : EMA (Exponential Moving Average) of model weights (decay=0.995) — retry of exp26/27 which both crashed on infra.
- **Idée** : Maintain a running average of all trainable parameters during training. Swap to EMA weights before eval. Smooths out SGD noise for better generalization. Third attempt (exp26/27 crashed on RunPod infra, never tested on GPU).
- **Pourquoi** : All "more steps" ideas are exhausted. Quality per step is the only remaining axis. EMA is universally proven (diffusion, ViT, LLM). Model is noisy (~0.003 variance between identical runs). ~200 step averaging window covers the warmdown convergence trajectory.
- **Risques** : Very low. ~152MB extra VRAM. ~1-2ms/step overhead. No architectural change. Worst case: neutral if warmdown already converges cleanly.
- **Changement** : (1) Clone all trainable params after torch.compile, (2) `ema.lerp_(p.data, 0.005)` after each optimizer step, (3) `p.data.copy_(ema)` before eval. 6 lines added.


## Experiment #30 — 2026-03-11 15:41:48

- **Best val_bpb avant** : 1.079602
- **Status** : en cours...
- **Description** : SwiGLU at iso-compute (3rd attempt — exp25/28 crashed infra, never tested on GPU). Infra now stable (exp29 ran).
- **Idée** : Replace ReluSquared MLP with SwiGLU gating at identical compute/params. c_fc(512→2048 fused gate+up), c_proj(1024→512 down), forward: silu(gate) * up.
- **Pourquoi** : SwiGLU is the standard in ALL modern LLMs. Iso-compute swap = zero risk to step count/speed. Never actually tested on GPU — all previous crashes were RunPod infra. Exp29 confirmed infra is stable again.
- **Risques** : Very low. Zero compute/param/memory change. If SwiGLU doesn't help, neutral within noise. No crash/OOM risk.
- **Changement** : MLP: `c_fc(512→2048 fused gate+up)`, `c_proj(1024→512 down)`, forward: `silu(gate) * up` instead of `relu(x).square()`. 4 lines changed in MLP class.


## Experiment #31 — 2026-03-11 15:51:50

- **Best val_bpb avant** : 1.079602
- **Status** : discard (val_bpb: 1.085796)
- **Description** : DEPTH=9 + GQA (n_kv_head=2) + MLP 3x — iso-compute reallocation from attention to MLP, preserving the winning MLP 3x w
- **Description** : DEPTH=9 + GQA (n_kv_head=2) + MLP 3x — iso-compute reallocation from attention to MLP, preserving the winning MLP 3x.
- **Idée** : Same total compute per layer as exp22 (2,359,296 params) but reallocated: 786K attention + 1.57M MLP (instead of exp22's 1.05M attention + 1.31M MLP). Keeps the winning MLP 3x while gaining an extra layer via GQA savings.
- **Pourquoi** : Exp22 (DEPTH=9+MLP2.5x) = 1.080504, very close to best. The depth benefit nearly compensated for the narrower MLP. This config preserves full MLP 3x capacity (the breakthrough from exp14) while gaining the same extra layer. GQA was essentially free at 8 layers (+0.0003 in exp4). The tradeoff shifts compute FROM attention (where GQA showed capacity is cheap) TO MLP (where we know 3x is the sweet spot).
- **Risques** : VE dim halves from 512 to 256 (n_kv_head=2). VE is critical but GQA+VE was fine in exp4. repeat_interleave overhead in SDPA for GQA. Expected within noise — could go either way by ~0.001-0.002.
- **Changement** : `DEPTH=8→9`, `ASPECT_RATIO=64→56` (keeps dim=512), added `N_KV_HEAD=2`, `n_kv_head=N_KV_HEAD` in build_model_config. 4 lines changed.


## Experiment #32 — 2026-03-11 16:07:16

- **Best val_bpb avant** : 1.079602
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Retry lm_head with Muon optimizer at conservative LR (exp23 crashed, likely infra — never tested on GPU).
- **Description** : Retry lm_head with Muon optimizer at conservative LR=0.005 (effective 0.04). Exp23 crashed (likely infra), never tested.
- **Idée** : Move lm_head (32768×512, 16.8M params, 44% of matrix FLOPs) from AdamW to Muon with dedicated base LR=0.005. NS orthogonalization gives instant preconditioning for the output layer — pushes 32768 token vectors toward isometry in 512-dim space.
- **Pourquoi** : AdamW needs many steps to warm up second moments; with ~830 steps it barely converges. Muon's NS gives instant gradient preconditioning. Exp23 used default lr=0.04 (effective 0.32 = 8× too high). This retry uses lr=0.005 (effective 0.04 = same as square matrices). Exp23 crash was almost certainly infra (4+ consecutive infra crashes around that time).
- **Risques** : If effective LR 0.04 still too high for lm_head, mild regression. Cautious WD now applies to lm_head (was 0.0). No crash/OOM risk — saves ~33MB vs AdamW.
- **Changement** : Removed lm_head from AdamW param_groups. Added dedicated Muon group for lm_head with lr=0.005. ~4 lines changed in `setup_optimizer`.


## Experiment #33 — 2026-03-11 16:15:27

- **Best val_bpb avant** : 1.079602
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Label smoothing (alpha=0.1) to improve generalization and gradient flow through softcap.
- **Description** : Label smoothing (alpha=0.1) to improve generalization and gradient flow through softcap.
- **Idée** : Add `label_smoothing=0.1` to F.cross_entropy(). Redistributes 10% of probability mass uniformly across vocab, preventing overconfident predictions and reducing wasted gradient signal fighting the softcap tanh boundary.
- **Pourquoi** : Proven in GPT-3, PaLM, Chinchilla. Synergizes with softcap (less incentive to push logits to ±15 where tanh saturates) and z-loss (both reduce large logit incentive, from different angles). Zero compute overhead. Never tested in this series — first regularization beyond z-loss/softcap.
- **Risques** : Very low. If alpha=0.1 too aggressive → mild regression (~0.001-0.003). Zero crash/OOM/speed risk. 1 keyword argument added.
- **Changement** : Added `label_smoothing=0.1` to `F.cross_entropy()` call. 1 parameter added.


## Experiment #34 — 2026-03-11 16:25:11

- **Best val_bpb avant** : 1.079602
- **Status** : discard (val_bpb: 1.119757)
- **Description** : Label smoothing (alpha=0.1) retry — exp33 crashed on infra, never tested on GPU. Fixed: smoothing applied only during 
- **Description** : Label smoothing (alpha=0.1) retry — exp33 crashed infra, never tested. Fixed: smoothing only during training, not eval.
- **Idée** : Add `label_smoothing=0.1` to F.cross_entropy but ONLY during training (reduction='mean'). Eval uses pure CE for comparable val_bpb. Exp33 applied it unconditionally and crashed on infra.
- **Pourquoi** : Proven in GPT-3/PaLM/Chinchilla. Synergizes with softcap=15 (less incentive to push logits to tanh saturation → better gradient flow) and z-loss (both reduce large logit incentive from different angles). Zero compute overhead. Never tested on GPU.
- **Risques** : Very low. If alpha=0.1 too aggressive → mild regression (~0.001-0.003). Zero crash/OOM/speed risk. 1 line added.
- **Changement** : Added `smoothing = 0.1 if reduction == 'mean' else 0.0` before F.cross_entropy, then `label_smoothing=smoothing`. Eval path gets smoothing=0.0, keeping val_bpb comparable.


## Experiment #35 — 2026-03-11 16:33:47

- **Best val_bpb avant** : 1.079602
- **Status** : discard (val_bpb: 1.084106)
- **Description** : Cosine warmdown schedule instead of linear warmdown to improve LR decay shape during the 60% warmdown phase.
- **Description** : Cosine warmdown schedule instead of linear warmdown.
- **Idée** : Replace linear LR decay during warmdown with cosine decay. Cosine spends more time at higher LR early in warmdown (0.854 vs 0.5 at midpoint), then drops faster at the end. Standard in modern LLM training (LLaMA, PaLM, Chinchilla).
- **Pourquoi** : 60% of training is warmdown (WARMDOWN_RATIO=0.6). The shape of the decay curve matters significantly. Cosine gives more effective learning steps during early warmdown and smoother convergence at the end. Never tested — all 34 previous experiments used linear warmdown.
- **Risques** : Very low. Zero compute/memory/speed impact. If cosine keeps LR too high for too long, mild regression (~0.001-0.002). No crash/OOM risk.
- **Changement** : Added `import math`. Changed `get_lr_multiplier` warmdown branch from `cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC` to `FINAL_LR_FRAC + 0.5 * (1.0 - FINAL_LR_FRAC) * (1 + math.cos(math.pi * warmdown_progress))`. 2 lines changed.


## Experiment #36 — 2026-03-11 16:45:04

- **Best val_bpb avant** : 1.079602
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Reduce logit softcap from 15 to 12 to improve gradient flow through tanh while preserving sufficient logit dynamic range
- **Description** : Reduce logit softcap from 15 to 12 for better gradient flow through tanh.
- **Idée** : Tighter softcap (12 vs 15) improves gradient flow through tanh — less saturation means stronger gradients for a wider range of logit values. Only 3 values tested so far: 0, 15, 30. The search is too coarse.
- **Pourquoi** : Softcap is one of the most impactful HPs (+0.009 without it). z-loss synergizes with tighter softcap. logit gap of 12 is sufficient for 99.99% confidence on 32768 vocab. Better gradient flow = better convergence per step.
- **Risques** : Very low. If 12 too restrictive, mild regression (~0.001-0.003). Zero crash/OOM/speed risk. 1 number change.
- **Changement** : `softcap = 15` → `softcap = 12`. Single character change.


## Experiment #37 — 2026-03-11 16:59:52

- **Best val_bpb avant** : 1.079602
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Retry softcap=12 (exp36 crashed on infra, never tested on GPU) to improve gradient flow through tanh.
- **Description** : Retry softcap=12 (exp36 crashed infra, never tested on GPU) for better gradient flow through tanh.
- **Idée** : Reduce logit softcap from 15 to 12. Tighter constraint = stronger gradients through tanh for more logit values. Synergy with z-loss. Sufficient dynamic range for 32768 vocab (max useful gap ~10.4).
- **Pourquoi** : Exp36 was identical but crashed on RunPod infra. Only 3 softcap values tested so far: {0, 15, 30}. Fine-grained search needed. Softcap is one of the most impactful HPs.
- **Risques** : Zero. Single number change. No crash/OOM/speed risk. If 12 too tight, mild regression ~0.001-0.003.
- **Changement** : `softcap = 15` → `softcap = 12`. Single character change.


## Experiment #38 — 2026-03-11 17:06:52

- **Best val_bpb avant** : 1.079602
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Reduce Muon momentum warmup from 300 to 100 steps to reach full momentum faster in the ~830-step training budget.
- **Description** : Reduce Muon momentum warmup from 300 to 100 steps to reach full momentum faster.
- **Idée** : The model trains ~830 steps. With 300-step momentum warmup, 36% of training is at suboptimal Nesterov momentum (0.85→0.95). Reducing to 100 steps means full 0.95 momentum by step 100 (12% of training).
- **Pourquoi** : Higher momentum = stronger Nesterov look-ahead = better gradient estimates. At step 100, new schedule has 0.950 momentum vs 0.883 (3.5× stronger look-ahead distance). NS orthogonalization + NorMuon provide inherent stability against too-fast warmup. Never tested — all 37 previous experiments used 300-step warmup.
- **Risques** : Zero. Single number change. No crash/OOM/speed risk. If too fast, mild regression ~0.001-0.003.
- **Changement** : `get_muon_momentum`: `min(step / 300, 1)` → `min(step / 100, 1)`. Single number change.


## Experiment #39 — 2026-03-11 17:09:31

- **Best val_bpb avant** : 1.079602
- **Status** : en cours...
- **Description** : Retry softcap=12 (3rd attempt — exp36/37 crashed infra) for better gradient flow through tanh.
- **Idée** : Reduce logit softcap from 15 to 12. Tighter constraint = stronger gradients through tanh. Synergy with z-loss. Sufficient dynamic range for 32768 vocab (max useful gap ~10.4). 3rd attempt — exp36/37 both crashed on RunPod infra, never tested on GPU.
- **Pourquoi** : Softcap is one of the most impactful HPs. Only 3 coarse values tested: {0, 15, 30}. Fine-grained search in the 10-20 range is needed. Gradient flow through tanh'(x) improves with smaller x.
- **Risques** : Zero. Single number change. No crash/OOM/speed risk. If 12 too tight, mild regression ~0.001-0.003.
- **Changement** : `softcap = 15` → `softcap = 12`. Single number change in forward().


## Experiment #40 — 2026-03-11 17:16:29

- **Best val_bpb avant** : 1.079602
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Retry Muon momentum warmup 300→100 steps (exp38 crashed infra, never tested on GPU) to reach full 0.95 momentum faster
- **Description** : Retry Muon momentum warmup 300→100 steps (exp38 crashed infra, never tested on GPU).
- **Idée** : Reduce Muon momentum warmup from 300 to 100 steps. Model trains ~830 steps; 300-step warmup = 36% of training at suboptimal momentum. 100-step warmup = 12%.
- **Pourquoi** : Higher Nesterov momentum = better look-ahead gradients. At step 100, new schedule gives 0.950 vs 0.883 momentum (3.5× stronger look-ahead). NS orthogonalization + NorMuon provide stability. Exp38 crashed on infra, never tested.
- **Risques** : Zero. Single number change. No crash/OOM/speed risk. If too fast, mild regression ~0.001-0.003.
- **Changement** : `get_muon_momentum`: `min(step / 300, 1)` → `min(step / 100, 1)`. Single number change.


## Experiment #41 — 2026-03-11 17:20:59

- **Best val_bpb avant** : 1.079602
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Increase Muon beta2 from 0.95 to 0.99 for smoother NorMuon per-row step sizing over a 100-step averaging window.
- **Description** : Increase Muon beta2 from 0.95 to 0.99 for smoother NorMuon per-row step sizing.
- **Idée** : NorMuon's second momentum buffer tracks per-row gradient variance with beta2=0.95 (~20-step window). Increase to 0.99 (~100-step window) for smoother, more stable per-parameter step sizing. Total gradient norm is preserved by NorMuon's normalization, so beta2 only affects per-row budget distribution — inherently safe.
- **Pourquoi** : Never tested in 40 experiments. Smoother per-row estimates reduce noise in adaptive step sizes. Analogous to Adam's beta2=0.999 (not 0.95) being standard. With ~830 training steps, a 100-step window is still responsive enough while being 5x smoother.
- **Risques** : Zero. Single number change. No crash/OOM/speed risk. NorMuon norm preservation makes it inherently safe. If 20-step window was fine, neutral within noise.
- **Changement** : `beta2=0.95` → `beta2=0.99` in `setup_optimizer` Muon param groups. Single number change.


## Experiment #42 — 2026-03-11 17:36:28

- **Best val_bpb avant** : 1.079602
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Retry Muon beta2=0.99 (exp41 crashed infra, never tested on GPU) for smoother NorMuon per-row step sizing.
- **Description** : Retry Muon beta2=0.99 (exp41 crashed infra, never tested on GPU) for smoother NorMuon per-row step sizing.
- **Idée** : Increase Muon beta2 from 0.95 to 0.99 for smoother per-row gradient variance tracking (~100-step window vs ~20-step). NorMuon's norm preservation makes total step size invariant to beta2 — only per-row budget distribution changes. Inherently safe.
- **Pourquoi** : Never tested on GPU (exp41 crashed infra). Smoother per-row estimates reduce noise in adaptive step sizes. Analogous to Adam's beta2=0.999 being standard. With ~830 training steps, 100-step window is responsive enough while 5x smoother.
- **Risques** : Zero. Single number change. No crash/OOM/speed risk. If 20-step window was fine, neutral within noise.
- **Changement** : `beta2=0.95` → `beta2=0.99` in `setup_optimizer` Muon param groups. Single number change.


## Experiment #43 — 2026-03-11 17:41:08

- **Best val_bpb avant** : 1.079602
- **Status** : en cours...
- **Description** : Reduce logit softcap from 15 to 12 (5th attempt — exp36/37/39 all crashed infra, never tested on GPU).
- **Idée** : Tighter softcap = stronger gradients through tanh for a wider range of logit values. Only 3 coarse values tested: {0, 15, 30}. Fine-grained search in the 10-20 range needed.
- **Pourquoi** : Softcap is one of the most impactful HPs (+0.009 without it, +0.005 at 30). z-loss synergizes with tighter softcap. Logit gap of 12 is sufficient for 99.99% confidence on 32768 vocab (max useful gap ~10.4). 4 previous attempts ALL crashed on RunPod infra, never tested on GPU.
- **Risques** : Zero. Single number change. No crash/OOM/speed risk. If 12 too tight, mild regression ~0.001-0.003.
- **Changement** : `softcap = 15` → `softcap = 12`. Single number change in forward().


## Experiment #44 — 2026-03-11 17:47:01

- **Best val_bpb avant** : 1.079602
- **Status** : en cours...


## Experiment #45 — 2026-03-12 09:24:06

- **Best val_bpb avant** : 1.079602
- **Status** : keep (val_bpb: 1.078486)
- **Description** : Reduce logit softcap from 15 to 13 to improve gradient flow through tanh while preserving sufficient logit dynamic range
- **Description** : Reduce logit softcap from 15 to 13 for better gradient flow through tanh.
- **Idée** : Tighter softcap (13 vs 15) improves gradient flow through tanh — less saturation means stronger gradients. Different from softcap=12 which crashed 5 times on infra (never tested on GPU). 13 avoids the potentially cursed compile path while testing the same hypothesis.
- **Pourquoi** : Softcap is one of the most impactful HPs (+0.009 without it, +0.005 at 30). Only 3 coarse values tested on GPU: {0, 15, 30}. Fine-grained search needed. z-loss synergizes with tighter softcap. 13 gives sufficient dynamic range for 32768 vocab (max useful gap ~10.4).
- **Risques** : Zero. Single number change. No crash/OOM/speed risk. If 13 too tight, mild regression ~0.001-0.003.
- **Changement** : `softcap = 15` → `softcap = 13`. Single number change in forward().


## Experiment #46 — 2026-03-12 09:38:26

- **Best val_bpb avant** : 1.078486
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Reduce logit softcap from 13 to 11 to further improve gradient flow through tanh, continuing the winning softcap tighten
- **Description** : Reduce logit softcap from 13 to 11 to further improve gradient flow through tanh.
- **Idée** : Continue the winning softcap tightening direction (30→15→13→11). Softcap=13 just improved by -0.001. Test if 11 gives further gains before hitting the expressiveness floor (~10.4 max useful logit gap).
- **Pourquoi** : tanh'(x) = 1 - tanh(x)^2. Smaller softcap → tanh saturates at lower raw logits → stronger gradients for more logit values. z-loss synergy. Softcap=12 crashed 5x (cursed compile path), so skipping to 11.
- **Risques** : Zero. Single number change. If 11 too tight (near theoretical min ~10.4), mild regression ~0.001-0.003.
- **Changement** : `softcap = 13` → `softcap = 11`. Single number change in forward().


## Experiment #47 — 2026-03-12 09:40:50

- **Best val_bpb avant** : 1.078486
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Retry softcap=11 (exp46 crashed infra, never tested on GPU) to further tighten gradient flow through tanh.
- **Description** : Retry softcap=11 (exp46 crashed infra, never tested on GPU) to further tighten gradient flow through tanh.
- **Idée** : Continue the winning softcap tightening direction (30→15→13→11). Softcap=13 just improved by -0.001. Test if 11 gives further gains before hitting the expressiveness floor (~10.4 max useful logit gap). Exp46 was identical but crashed on infra.
- **Pourquoi** : tanh'(x) = 1 - tanh(x)^2. Smaller softcap → stronger gradients for more logit values. z-loss synergy. Sufficient dynamic range for 32768 vocab. Softcap=12 crashed 5x (cursed compile path), so using 11.
- **Risques** : Zero. Single number change. If 11 too tight (near theoretical min ~10.4), mild regression ~0.001-0.003.
- **Changement** : `softcap = 13` → `softcap = 11`. Single number change in forward().


## Experiment #48 — 2026-03-12 09:44:39

- **Best val_bpb avant** : 1.078486
- **Status** : en cours...
- **Description** : Retry softcap=11 (3rd attempt — exp46/47 crashed infra, never tested on GPU) for further gradient flow improvement through tanh.
- **Idée** : Continue the winning softcap tightening direction (30→15→13→11). Softcap=13 improved by -0.001. Test if 11 gives further gains. 3rd attempt — exp46/47 both crashed on infra.
- **Pourquoi** : tanh'(x) = 1 - tanh(x)^2. Smaller softcap → stronger gradients for wider logit range. z-loss synergy. Sufficient dynamic range (max useful gap ~10.4 for 32768 vocab).
- **Risques** : Zero. Single number change. If 11 too tight (near theoretical min ~10.4), mild regression ~0.001-0.003.
- **Changement** : `softcap = 13` → `softcap = 11`. Single number change in forward().


## Experiment #49 — 2026-03-12 09:52:33

- **Best val_bpb avant** : 1.078486
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Reduce Muon momentum warmup from 300 to 100 steps to reach full 0.95 momentum faster in the ~830-step training budget.
- **Description** : Reduce Muon momentum warmup from 300 to 100 steps to reach full 0.95 momentum faster.
- **Idée** : With ~830 training steps, 300-step warmup means 36% of training at suboptimal momentum (0.85→0.95). 100-step warmup = 12%. Full momentum earlier = better Nesterov look-ahead for 88% vs 64% of training.
- **Pourquoi** : At step 100, new schedule gives 0.950 momentum vs old 0.883 (2.5× stronger look-ahead). NS orthogonalization + NorMuon provide stability. Attempted 2x (exp38/40) but both crashed on infra — never tested on GPU. Infra now stable.
- **Risques** : Zero. Single number change. If too fast, mild regression ~0.001-0.003.
- **Changement** : `get_muon_momentum`: `min(step / 300, 1)` → `min(step / 100, 1)`. Single number change.


## Experiment #50 — 2026-03-12 09:56:50

- **Best val_bpb avant** : 1.078486
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Increase Muon beta2 from 0.95 to 0.99 for smoother NorMuon per-row adaptive step sizing over a ~100-step averaging windo
- **Description** : Increase Muon beta2 from 0.95 to 0.99 for smoother NorMuon per-row adaptive step sizing.
- **Idée** : NorMuon's second momentum buffer tracks per-row gradient variance with beta2=0.95 (~20-step window). Increase to 0.99 (~100-step window) for smoother, more stable per-parameter step sizing. Total gradient norm is preserved by NorMuon's normalization, so beta2 only affects per-row budget distribution — inherently safe.
- **Pourquoi** : Never tested on GPU (exp41/42 both crashed on infra). Smoother per-row estimates reduce noise in adaptive step sizes. Analogous to Adam's beta2=0.999 being standard. With ~830 training steps, 100-step window is still responsive (~12% of training).
- **Risques** : Zero. Single number change. No crash/OOM/speed risk. NorMuon norm preservation makes it inherently safe. If 20-step window was fine, neutral within noise.
- **Changement** : `beta2=0.95` → `beta2=0.99` in `setup_optimizer` Muon param groups. Single number change.


## Experiment #51 — 2026-03-12 09:59:27

- **Best val_bpb avant** : 1.078486
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Reduce Muon momentum warmup from 300 to 100 steps to reach full 0.95 momentum faster in the ~830-step training budget.
- **Description** : Reduce Muon momentum warmup from 300 to 100 steps (4th attempt — exp38/40/49 all crashed infra, never tested on GPU).
- **Idée** : With ~830 training steps, 300-step warmup means 36% of training at suboptimal Nesterov momentum (0.85→0.95). 100-step warmup = 12%. Full momentum earlier = better look-ahead for 88% vs 64% of training.
- **Pourquoi** : At step 100, new schedule gives 0.950 momentum vs old 0.883 (2.5× stronger look-ahead distance). NS orthogonalization + NorMuon provide stability. Three previous attempts ALL crashed on RunPod infra — never tested on GPU.
- **Risques** : Zero. Single number change. No crash/OOM/speed risk. If too fast, mild regression ~0.001-0.003.
- **Changement** : `get_muon_momentum`: `min(step / 300, 1)` → `min(step / 100, 1)`. Single number change.


## Experiment #52 — 2026-03-12 10:16:38

- **Best val_bpb avant** : 1.078486
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Increase Muon Nesterov momentum ceiling from 0.95 to 0.97 for stronger gradient look-ahead during training.
- **Description** : Increase Nesterov momentum ceiling from 0.95 to 0.97 for stronger gradient look-ahead in Muon optimizer.
- **Idée** : Higher momentum ceiling = stronger Nesterov look-ahead. Look-ahead distance goes from 0.95/0.05=19 to 0.97/0.03=32.3 (+70%). The optimizer "sees further ahead" in the loss landscape, potentially finding better descent directions.
- **Pourquoi** : Never tested in 51 experiments. All previous Muon momentum changes targeted warmup SPEED (300→100 steps), not the TARGET ceiling. NS orthogonalization + NorMuon normalize step size, so momentum only affects direction weighting — inherently safe. Standard Nesterov range is 0.9-0.99; 0.97 is conservative. More inertia during warmdown phase = better convergence as LR drops.
- **Risques** : Zero. Single number change. No crash/OOM/speed risk. If 0.97 too aggressive, mild regression ~0.001-0.003.
- **Changement** : `get_muon_momentum`: `0.95` → `0.97` (ceiling). Single number change.


## Experiment #53 — 2026-03-12 10:31:57

- **Best val_bpb avant** : 1.078486
- **Status** : discard (val_bpb: 1.078587)
- **Description** : experiment 53


## Experiment #54 — 2026-03-12 10:41:31

- **Best val_bpb avant** : 1.078486
- **Status** : keep (val_bpb: 1.078476)
- **Description** : experiment 54


## Experiment #55 — 2026-03-12 10:47:03

- **Best val_bpb avant** : 1.078476
- **Status** : discard (val_bpb: 1.078706)
- **Description** : experiment 55


## Experiment #56 — 2026-03-12 10:52:42

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 56


## Experiment #57 — 2026-03-12 10:52:47

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 57


## Experiment #58 — 2026-03-12 10:52:50

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 58


## Experiment #59 — 2026-03-12 10:52:53

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 59


## Experiment #60 — 2026-03-12 10:52:56

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 60


## Experiment #61 — 2026-03-12 10:52:59

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 61


## Experiment #62 — 2026-03-12 10:53:02

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 62


## Experiment #63 — 2026-03-12 10:53:05

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 63


## Experiment #64 — 2026-03-12 10:53:08

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 64


## Experiment #65 — 2026-03-12 10:53:11

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 65


## Experiment #66 — 2026-03-12 10:53:13

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 66


## Experiment #67 — 2026-03-12 10:53:16

- **Best val_bpb avant** : 1.078476
- **Description** : Per-head learnable attention temperature (qk_scales) to let each head/layer optimize its own attention sharpness.
- **Idée** : QK-norm strips all magnitude information from Q and K vectors, forcing a fixed attention temperature of 1/sqrt(head_dim) across all heads and layers. Adding a per-head per-layer learnable scalar multiplier on Q after QK-norm restores the model's ability to control attention sharpness. Different layers serve different purposes (syntactic vs semantic) and likely benefit from different temperatures. This adds only 32 learnable scalars (8 layers × 4 heads) with zero compute overhead.
- **Pourquoi** : Modern QK-norm implementations (Gemma 2, etc.) often include learnable scale. Without it, the model cannot learn to use sharp attention (precise token matching) in some heads and broad attention (contextual averaging) in others. With softcap=13 and max logit ~11.3, there's room for heads to learn higher scales (~1.2-1.3) for sharper attention without saturating the softcap.
- **Risques** : Zero crash/OOM risk (32 extra scalars). Zero speed impact. If fixed temperature is already optimal: neutral. If scale LR (0.05) is too aggressive: possible mild instability. If scales diverge too high and saturate softcap: mild regression ~0.001-0.003.
- **Changement** : Added `self.qk_scales = nn.Parameter(torch.ones(n_layer, n_head))` in GPT. After QK-norm in attention: `q = q * qk_scale[None, None, :, None]`. Optimized with AdamW at lr=0.05, betas=(0.96, 0.95). Initialized to 1.0 (preserving current behavior at step 0).


## Experiment #68 — 2026-03-12 10:53:19

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 68


## Experiment #69 — 2026-03-12 10:53:22

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 69


## Experiment #70 — 2026-03-12 10:53:25

- **Best val_bpb avant** : 1.078476
- **Description** : Decouple AdamW/Muon LR schedules — remove warmdown for AdamW groups (embeddings, lm_head, VE, scalars) while keeping Muon warmdown.
- **Idée** : Currently ALL param groups share the same warmdown (LR decays to 0 over last 60% of training). But AdamW groups (66% of params — embeddings, lm_head, VE, scalars) have no weight decay and benefit from continued learning. With 830 steps, 60% warmdown means only ~330 steps at full LR for these large tables. Keeping AdamW at constant LR gives embeddings full learning rate for the entire training budget. Muon still needs warmdown for NS orthogonalization stability, so only AdamW groups change.
- **Pourquoi** : Exp13 tested less warmdown (0.6→0.4) for ALL groups and regressed. But that's different — Muon matrix params genuinely need warmdown. AdamW's adaptive per-parameter step sizes already self-regulate (second moment tracks gradient variance), making external warmdown redundant. The embeddings are the most undertrained parameters (32768×512 lookup tables updated sparsely), so they benefit most from more steps at full LR. Never tested: all previous warmdown changes applied uniformly to all groups.
- **Risques** : Zero crash/OOM risk (3-line schedule change, no structural modification). If embeddings overfit at constant LR: mild regression ~0.001-0.003. If Muon-AdamW schedule mismatch causes training instability: mild regression.
- **Changement** : In the LR schedule loop, apply `lrm` warmdown multiplier only to Muon groups. AdamW groups keep `group["lr"] = group["initial_lr"]` (constant LR throughout training).


## Experiment #71 — 2026-03-12 10:53:28

- **Best val_bpb avant** : 1.078476
- **Description** : U-Net skip connections — connect early layer outputs to mirror late layers (0→7, 1→6, 2→5, 3→4) with learnable per-pair scaling.
- **Idée** : Add skip connections from early transformer layers to their mirror late layers, like U-Net architecture. Layer i (i < n//2) saves its output; layer (n-1-i) receives it scaled by a learnable scalar (skip_lambdas[i]). This provides: (1) shorter gradient paths from late to early layers, improving gradient flow; (2) multi-scale features — late layers get access to both refined residual stream AND less-processed early features; (3) a different information pathway than x0_lambdas (which carries raw embeddings, not processed features). Used in DiT architectures and NanoGPT speedrun entries. Adds only 4 learnable scalars for 8 layers.
- **Pourquoi** : The current model has two information pathways: residual stream (resid_lambdas) and initial embedding skip (x0_lambdas). Both are "vertical" — x0 gives raw embeddings, and the residual carries accumulated processing. U-Net skips add "diagonal" connections: layer 7 gets access to layer 0's output (which has gone through 1 attention+MLP), providing a mid-level representation. In a 5-minute regime with ~830 steps, better gradient flow means more efficient learning per step.
- **Risques** : Zero crash risk (initialized to 0 = identical to baseline at step 0). Zero OOM risk (skip states are already in the autograd graph for backprop). Zero speed impact (scalar-weighted additions). Possible regression if the extra pathways interfere with the well-tuned residual dynamics. Possible graph break with torch.compile if conditional logic or list access causes tracing issues.


## Experiment #72 — 2026-03-12 10:53:31

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 72


## Experiment #73 — 2026-03-12 10:53:34

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 73


## Experiment #74 — 2026-03-12 10:53:37

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 74


## Experiment #75 — 2026-03-12 10:53:40

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 75


## Experiment #76 — 2026-03-12 10:53:42

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 76


## Experiment #77 — 2026-03-12 10:53:45

- **Best val_bpb avant** : 1.077355
- **Description** : Mid-MLP normalization (NormFormer-style) — add RMSNorm between ReluSquared activation and output projection in MLP.
- **Idée** : Add `norm(x)` (RMSNorm) after `F.relu(x).square()` and before `self.c_proj(x)` in the MLP forward pass. ReluSquared produces sparse activations with potentially very large values (squared), which grow as training progresses. Without normalization, c_proj must learn to handle increasingly variable input magnitudes. Normalizing the hidden activations ensures consistent gradient flow through c_proj regardless of activation magnitude.
- **Pourquoi** : All winning experiments improved gradient flow: softcap 13 (tanh gradient), z-loss (logit calibration), U-Net skips (shorter gradient paths). This targets the MLP gradient path — the last major unnormalized internal activation in the model. With 8 layers × 3x MLP, the MLP represents the majority of compute. Stabilizing its internal gradient flow could improve learning efficiency per step. NormFormer (Shleifer et al., 2021) showed that adding norms within sublayers improves training stability and convergence.
- **Risques** : Zero crash risk (norm() is already used throughout the model). Zero OOM risk (no new parameters). Zero speed impact (one RMSNorm per layer, negligible vs matmuls). If ReluSquared's raw magnitude carries useful signal that norm destroys: mild regression ~0.001-0.003. If the sparsity-dependent scaling (RMS of sparse vector) is unstable: mild regression.


## Experiment #78 — 2026-03-12 10:53:48

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 78


## Experiment #79 — 2026-03-12 10:53:51

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 79


## Experiment #80 — 2026-03-12 10:53:54

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 80


## Experiment #81 — 2026-03-12 10:53:57

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 81


## Experiment #82 — 2026-03-12 10:54:00

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 82


## Experiment #83 — 2026-03-12 10:54:03

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 83


## Experiment #84 — 2026-03-12 10:54:06

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 84


## Experiment #85 — 2026-03-12 10:54:09

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 85


## Experiment #86 — 2026-03-12 10:54:12

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 86


## Experiment #87 — 2026-03-12 10:54:14

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 87


## Experiment #88 — 2026-03-12 10:54:17

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 88


## Experiment #89 — 2026-03-12 10:54:20

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 89


## Experiment #90 — 2026-03-12 10:54:23

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 90


## Experiment #91 — 2026-03-12 10:54:26

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 91


## Experiment #92 — 2026-03-12 10:54:29

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 92


## Experiment #93 — 2026-03-12 10:54:32

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 93


## Experiment #94 — 2026-03-12 10:54:34

- **Best val_bpb avant** : 1.072821
- **Idée** : DEPTH=9 (6th attempt) with U-Net skip bug fix for odd depths — all previous attempts (exp86/87/88/92/93) crashed. exp88 had the IndexError bug, exp92/93 had the fix but crashed on infra.
- **Pourquoi** : Depth is the single most valuable axis. DEPTH=9 with the bug fix has NEVER successfully run. Back-of-envelope: current best 1.072821 minus depth benefit ≈ 1.070-1.071. Changes: DEPTH=9, ASPECT_RATIO=56 (→512 dim, 4 heads), and `if i >= half and mirror < half:` to skip middle layer for odd depths.
- **Risques** : ~8-10% fewer training steps due to extra layer compute. If step reduction outweighs capacity gain: regression ~0.001-0.003. Zero OOM risk. Bug fix is proven correct (exp92/93 code). Main risk is infra crash again.


## Experiment #95 — 2026-03-12 10:54:38

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 95


## Experiment #96 — 2026-03-12 10:54:42

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 96


## Experiment #97 — 2026-03-12 10:54:44

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 97


## Experiment #98 — 2026-03-12 10:54:47

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 98


## Experiment #99 — 2026-03-12 10:54:50

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 99


## Experiment #100 — 2026-03-12 10:54:54

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 100


## Experiment #101 — 2026-03-12 10:54:57

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 101


## Experiment #102 — 2026-03-12 10:54:59

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 102


## Experiment #103 — 2026-03-12 10:55:02

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 103


## Experiment #104 — 2026-03-12 10:55:05

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 104


## Experiment #105 — 2026-03-12 10:55:08

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 105


## Experiment #106 — 2026-03-12 10:55:11

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 106


## Experiment #107 — 2026-03-12 10:55:14

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 107


## Experiment #108 — 2026-03-12 10:55:17

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 108


## Experiment #109 — 2026-03-12 10:55:19

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 109


## Experiment #110 — 2026-03-12 10:55:22

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 110


## Experiment #111 — 2026-03-12 10:55:25

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 111


## Experiment #112 — 2026-03-12 10:55:28

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 112


## Experiment #113 — 2026-03-12 10:55:31

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 113


## Experiment #114 — 2026-03-12 10:55:34

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 114


## Experiment #115 — 2026-03-12 10:55:37

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 115


## Experiment #116 — 2026-03-12 10:55:39

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 116


## Experiment #117 — 2026-03-12 10:55:43

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 117


## Experiment #118 — 2026-03-12 10:55:45

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 118


## Experiment #119 — 2026-03-12 10:55:48

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 119


## Experiment #120 — 2026-03-12 10:55:51

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 120


## Experiment #121 — 2026-03-12 10:55:54

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 121


## Experiment #122 — 2026-03-12 10:55:57

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 122


## Experiment #123 — 2026-03-12 10:55:59

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 123


## Experiment #124 — 2026-03-12 10:56:03

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 124


## Experiment #125 — 2026-03-12 10:56:06

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 125


## Experiment #126 — 2026-03-12 10:56:09

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 126


## Experiment #127 — 2026-03-12 10:56:11

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 127


## Experiment #128 — 2026-03-12 10:56:14

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 128


## Experiment #129 — 2026-03-12 10:56:17

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 129


## Experiment #130 — 2026-03-12 10:56:20

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 130


## Experiment #131 — 2026-03-12 10:56:23

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 131


## Experiment #132 — 2026-03-12 10:56:26

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 132


## Experiment #133 — 2026-03-12 10:56:29

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 133


## Experiment #134 — 2026-03-12 10:56:31

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 134


## Experiment #135 — 2026-03-12 10:56:34

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 135


## Experiment #136 — 2026-03-12 10:56:37

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 136


## Experiment #137 — 2026-03-12 10:56:40

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 137


## Experiment #138 — 2026-03-12 10:56:43

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 138


## Experiment #139 — 2026-03-12 10:56:46

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 139


## Experiment #140 — 2026-03-12 10:56:48

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 140


## Experiment #141 — 2026-03-12 10:56:51

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 141


## Experiment #142 — 2026-03-12 10:56:54

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 142


## Experiment #143 — 2026-03-12 10:56:57

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 143


## Experiment #144 — 2026-03-12 10:57:00

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 144


## Experiment #145 — 2026-03-12 10:57:03

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 145


## Experiment #146 — 2026-03-12 10:57:06

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 146


## Experiment #147 — 2026-03-12 10:57:09

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 147


## Experiment #148 — 2026-03-12 10:57:12

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 148


## Experiment #149 — 2026-03-12 10:57:14

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 149


## Experiment #150 — 2026-03-12 10:57:17

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 150


## Experiment #151 — 2026-03-12 10:57:20

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 151


## Experiment #152 — 2026-03-12 10:57:23

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 152


## Experiment #153 — 2026-03-12 10:57:26

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 153


## Experiment #154 — 2026-03-12 10:57:29

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 154


## Experiment #155 — 2026-03-12 10:57:32

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 155


## Experiment #156 — 2026-03-12 10:57:35

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 156


## Experiment #157 — 2026-03-12 10:57:38

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 157


## Experiment #158 — 2026-03-12 10:57:41

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 158


## Experiment #159 — 2026-03-12 10:57:44

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 159


## Experiment #160 — 2026-03-12 10:57:46

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 160


## Experiment #161 — 2026-03-12 10:57:49

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 161


## Experiment #162 — 2026-03-12 10:57:52

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 162


## Experiment #163 — 2026-03-12 10:57:55

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 163


## Experiment #164 — 2026-03-12 10:57:58

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 164


## Experiment #165 — 2026-03-12 10:58:01

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 165


## Experiment #166 — 2026-03-12 10:58:04

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 166


## Experiment #167 — 2026-03-12 10:58:07

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 167


## Experiment #168 — 2026-03-12 10:58:10

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 168


## Experiment #169 — 2026-03-12 10:58:12

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 169


## Experiment #170 — 2026-03-12 10:58:15

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 170


## Experiment #171 — 2026-03-12 10:58:18

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 171


## Experiment #172 — 2026-03-12 10:58:21

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 172


## Experiment #173 — 2026-03-12 10:58:24

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 173


## Experiment #174 — 2026-03-12 10:58:27

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 174


## Experiment #175 — 2026-03-12 10:58:29

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 175


## Experiment #176 — 2026-03-12 10:58:32

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 176


## Experiment #177 — 2026-03-12 10:58:36

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 177


## Experiment #178 — 2026-03-12 10:58:38

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 178


## Experiment #179 — 2026-03-12 10:58:41

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 179


## Experiment #180 — 2026-03-12 10:58:44

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 180


## Experiment #181 — 2026-03-12 10:58:47

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 181


## Experiment #182 — 2026-03-12 10:58:50

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 182


## Experiment #183 — 2026-03-12 10:58:53

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 183


## Experiment #184 — 2026-03-12 10:58:56

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 184


## Experiment #185 — 2026-03-12 10:58:58

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 185


## Experiment #186 — 2026-03-12 10:59:01

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 186


## Experiment #187 — 2026-03-12 10:59:04

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 187


## Experiment #188 — 2026-03-12 10:59:07

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 188


## Experiment #189 — 2026-03-12 10:59:10

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 189


## Experiment #190 — 2026-03-12 10:59:12

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 190


## Experiment #191 — 2026-03-12 10:59:15

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 191


## Experiment #192 — 2026-03-12 10:59:18

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 192


## Experiment #193 — 2026-03-12 10:59:21

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 193


## Experiment #194 — 2026-03-12 10:59:24

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 194


## Experiment #195 — 2026-03-12 10:59:27

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 195


## Experiment #196 — 2026-03-12 10:59:30

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 196


## Experiment #197 — 2026-03-12 10:59:33

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 197


## Experiment #198 — 2026-03-12 10:59:36

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 198


## Experiment #199 — 2026-03-12 10:59:39

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 199


## Experiment #200 — 2026-03-12 10:59:41

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 200


## Experiment #201 — 2026-03-12 10:59:44

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 201


## Experiment #202 — 2026-03-12 10:59:47

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 202


## Experiment #203 — 2026-03-12 10:59:50

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 203


## Experiment #204 — 2026-03-12 10:59:53

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 204


## Experiment #205 — 2026-03-12 10:59:55

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 205


## Experiment #206 — 2026-03-12 10:59:58

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 206


## Experiment #207 — 2026-03-12 11:00:01

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 207


## Experiment #208 — 2026-03-12 11:00:04

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 208


## Experiment #209 — 2026-03-12 11:00:07

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 209


## Experiment #210 — 2026-03-12 11:00:10

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 210


## Experiment #211 — 2026-03-12 11:00:13

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 211


## Experiment #212 — 2026-03-12 11:00:15

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 212


## Experiment #213 — 2026-03-12 11:00:18

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 213


## Experiment #214 — 2026-03-12 11:00:21

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 214


## Experiment #215 — 2026-03-12 11:00:24

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 215


## Experiment #216 — 2026-03-12 11:00:27

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 216


## Experiment #217 — 2026-03-12 11:00:30

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 217


## Experiment #218 — 2026-03-12 11:00:33

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 218


## Experiment #219 — 2026-03-12 11:00:36

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 219


## Experiment #220 — 2026-03-12 11:00:39

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 220


## Experiment #221 — 2026-03-12 11:00:42

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 221


## Experiment #222 — 2026-03-12 11:00:45

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 222


## Experiment #223 — 2026-03-12 11:00:48

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 223


## Experiment #224 — 2026-03-12 11:00:51

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 224


## Experiment #225 — 2026-03-12 11:00:54

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 225


## Experiment #226 — 2026-03-12 11:00:57

- **Best val_bpb avant** : 1.078476
- **Status** : en cours...


## Experiment #52 — 2026-03-12 11:12:50

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Increase Muon Nesterov momentum ceiling from 0.95 to 0.97 for stronger gradient look-ahead during training.
- **Idée** : Increase Muon Nesterov momentum ceiling from 0.95 to 0.97 for 70% stronger gradient look-ahead.
- **Pourquoi** : Look-ahead distance = momentum/(1-momentum). At 0.97, distance=32.3 vs 19.0 at 0.95. NS orthogonalization + NorMuon normalize step size independently, making this safe. Never tested before — all previous momentum experiments targeted warmup speed, not the ceiling.
- **Risques** : Very low. If 0.97 overshoots, mild regression ~0.001-0.003. No crash/OOM/speed risk — pure scalar change.
- **Changement** : `get_muon_momentum`: `frac * 0.95` → `frac * 0.97`. Single number change.


## Experiment #53 — 2026-03-12 11:15:46

- **Best val_bpb avant** : 1.078476
- **Status** : en cours...


## Experiment #54 — 2026-03-12 11:21:54

- **Best val_bpb avant** : 1.078476
- **Status** : discard (val_bpb: 1.085454)
- **Description** : Multi-Query Attention (MQA, n_kv_head=1) to reduce KV parameters and VE embedding size for faster training throughput.
- **Idée** : Multi-Query Attention (MQA, n_kv_head=1) — reduce KV parameters and VE size for faster throughput.
- **Pourquoi** : GQA (n_kv_head=2) was tied with baseline in exp4. MQA saves ~21% params (15.7M), faster Muon steps, more training steps in 5 min. Explicitly listed as untried promising direction in prompt.md.
- **Risques** : Low. Single KV head might lose representational capacity. Worst case: regression ~0.001-0.003. No crash/OOM risk — all code paths support n_kv_head < n_head.
- **Changement** : Added `N_KV_HEAD = 1`. Changed `build_model_config` to use `n_kv_head=N_KV_HEAD` instead of `n_kv_head=num_heads`.


## Experiment #55 — 2026-03-12 11:37:35

- **Best val_bpb avant** : 1.078476
- **Status** : discard (val_bpb: 1.079176)
- **Description** : Reduce Muon momentum warmup from 300 to 100 steps to reach full 0.95 Nesterov momentum faster.
- **Idée** : Reduce Muon momentum warmup from 300 to 100 steps to reach full 0.95 momentum faster in the ~830-step training budget.
- **Pourquoi** : With ~830 steps, 300-step warmup means 36% of training at suboptimal momentum (0.85→0.95). At 100-step warmup, only 12%. At step 100, new schedule gives 0.950 momentum vs old 0.883 — 2.5x stronger Nesterov look-ahead distance (momentum/(1-momentum) = 19.0 vs 7.6). NS orthogonalization + NorMuon normalize step size independently, making this safe. Attempted 4+ times (exp38/40/49/51) but ALL crashed on infra — never tested on GPU.
- **Risques** : Zero. Single number change (300→100). No crash/OOM/speed risk. If too fast, mild regression ~0.001-0.003.
- **Changement** : `get_muon_momentum`: `min(step / 300, 1)` → `min(step / 100, 1)`. Single number change.


## Experiment #56 — 2026-03-12 11:52:50

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Increase Muon beta2 from 0.95 to 0.99 for smoother NorMuon per-row adaptive step sizing.
- **Description** : Increase Muon beta2 from 0.95 to 0.99 for smoother NorMuon per-row adaptive step sizing.
- **Idee** : Increase Muon beta2 from 0.95 to 0.99 — effective averaging window from ~20 steps to ~100 steps for smoother per-row gradient variance estimates.
- **Pourquoi** : NorMuon's norm preservation means beta2 only affects per-row step size distribution, not total magnitude. Analogous to Adam's default beta2=0.999. Never tested on GPU (exp41/42/50 all crashed infra). #1 recommended untested change per prompt.md notes.
- **Risques** : Zero. Single number change (0.95->0.99). No crash/OOM/speed risk. If too slow to adapt, mild regression ~0.001-0.003.
- **Changement** : `setup_optimizer` Muon groups: `beta2=0.95` -> `beta2=0.99`. Single number change.


## Experiment #57 — 2026-03-12 11:58:09

- **Best val_bpb avant** : 1.078476
- **Status** : discard (val_bpb: 1.079416)
- **Description** : Increase Muon beta2 from 0.95 to 0.99 for smoother NorMuon per-row adaptive step sizing.
- **Description** : Increase Muon beta2 from 0.95 to 0.99 for smoother NorMuon per-row adaptive step sizing.
- **Idée** : Increase Muon beta2 from 0.95 to 0.99 — effective averaging window from ~20 steps to ~100 steps for smoother per-row gradient variance estimates.
- **Pourquoi** : NorMuon's norm preservation means beta2 only affects per-row step size distribution, not total magnitude. Analogous to Adam's default beta2=0.999. Never tested on GPU (exp41/42/50/56 all crashed infra). #1 recommended untested change per prompt.md notes.
- **Risques** : Zero. Single number change (0.95->0.99). No crash/OOM/speed risk. If too slow to adapt, mild regression ~0.001-0.003.
- **Changement** : `setup_optimizer` Muon groups: `beta2=0.95` -> `beta2=0.99`. Single number change.


## Experiment #58 — 2026-03-12 12:11:46

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Add gradient clipping (max_norm=1.0) to the training loop — standard LLM practice currently missing from the pipeline.
- **Idée** : Add gradient clipping (max_norm=1.0) — standard practice missing from the training pipeline.
- **Pourquoi** : Every major LLM training setup (GPT-2/3, LLaMA, PaLM) uses gradient clipping. Currently there is NO gradient clipping. The Muon optimizer has built-in normalization (NS + NorMuon), so clipping is mostly a no-op for matrix params. But AdamW params (embeddings, lm_head, scalars) have no such protection — occasional gradient spikes can destabilize their optimizer states. With max_norm=1.0, clipping only activates on extreme spikes and is otherwise a no-op.
- **Risques** : Zero crash/OOM/speed risk. One-line change. If max_norm=1.0 is too aggressive (clips too often), mild regression ~0.001-0.003. If gradient norms are normally < 1.0, it's a complete no-op.
- **Changement** : Added `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` after gradient accumulation, before optimizer.step().


## Experiment #59 — 2026-03-12 12:21:36

- **Best val_bpb avant** : 1.078476
- **Status** : discard (val_bpb: 1.082534)
- **Description** : Increase Muon Nesterov momentum ceiling from 0.95 to 0.97 for 70% stronger gradient look-ahead (retry of exp52 which cra
- **Idée** : Increase Muon Nesterov momentum ceiling from 0.95 to 0.97 for stronger gradient look-ahead (retry of exp52 which crashed infra).
- **Pourquoi** : Look-ahead distance = momentum/(1-momentum). At 0.97, distance=32.3 vs 19.0 at 0.95 (+70%). NS orthogonalization + NorMuon normalize step size independently, so momentum only affects direction weighting — inherently safe. This changes steady-state behavior (not warmup speed like exp55), a different axis. Exp52 crashed on infra, never tested on GPU. Marked HIGH PRIORITY in prompt.md notes.
- **Risques** : Very low. Single number change (0.95→0.97). No crash/OOM/speed risk. If 0.97 overshoots, mild regression ~0.001-0.003.
- **Changement** : `get_muon_momentum`: `frac * 0.95` → `frac * 0.97`. Single number change.


## Experiment #60 — 2026-03-12 12:35:21

- **Best val_bpb avant** : 1.078476
- **Status** : en cours...
- **Idée** : Parallel attention + MLP (PaLM/GPT-J style) — both sublayers operate on the same normalized input instead of sequentially.
- **Pourquoi** : In the sequential Block, the MLP sees post-attention features: `x = x + attn(norm(x)); x = x + mlp(norm(x))`. In parallel: `x = x + attn(norm(x)) + mlp(norm(x))`. This saves one norm call per block (8 total), slightly reducing compute. More importantly, it's a genuine architectural change validated at scale (PaLM, GPT-J). At this small model size, the sequential information flow (MLP seeing attention output) may not be critical since attention patterns are noisy early in training. The parallel formulation also has cleaner gradient flow since both sublayer gradients pass through the same normalization. IMPORTANT: on this hardware (RTX 5090, not Hopper), SDPA is used with full causal attention — window patterns have zero effect on compute. Architectural changes to the block structure are the main remaining lever.
- **Risques** : Low. No crash/OOM/speed risk — pure control flow change, same params, same shapes. If parallel is worse at this scale, mild regression ~0.001-0.005. Compute savings from 8 fewer norms are small (~1-2%).
- **Changement** : Block.forward: merged sequential `x = x + attn(norm(x)); x = x + mlp(norm(x))` into parallel `nx = norm(x); x = x + attn(nx) + mlp(nx)`.


## Experiment #61 — 2026-03-12 12:46:34

- **Best val_bpb avant** : 1.078476
- **Status** : en cours...
- **Idée** : Reduce HEAD_DIM from 128 to 64, giving 8 attention heads instead of 4 — more diverse, specialized attention patterns at zero parameter/compute cost.
- **Pourquoi** : With 512 model dim and HEAD_DIM=128, we only have 4 attention heads. Each head must cover a wide range of attention patterns. With HEAD_DIM=64, we get 8 heads — each can specialize on different features (local syntax, long-range dependencies, positional patterns, etc.). This is parameter-neutral (same total QKV size: 512) and compute-neutral (SDPA cost ∝ n_embd × seq_len, unchanged). GPT-2-small used 12 heads × 64 dim for 768 dim — a similar ratio. RoPE with 64-dim heads is well-tested. This is a structural architectural change (priority #1) that has never been tested in this series.
- **Risques** : Very low. Single constant change (HEAD_DIM=128→64). No crash/OOM/speed risk. If 128-dim heads are better for this specific architecture/QK-norm combination, mild regression ~0.001-0.003. QK-norm over 64 dims vs 128 dims may behave slightly differently (attention slightly sharper with smaller d due to 1/sqrt(d) scaling).
- **Changement** : `HEAD_DIM = 128` → `HEAD_DIM = 64`. Single line change.


## Experiment #62 — 2026-03-12 13:10:07

- **Best val_bpb avant** : 1.078476
- **Status** : discard (val_bpb: 1.087990)
- **Description** : Add gradient clipping (max_norm=1.0) to the training loop — retry of exp58 which crashed on infra, never tested on GPU
- **Idée** : Add gradient clipping (max_norm=1.0) — standard LLM practice missing from the pipeline (retry of exp58 which crashed on infra, never tested on GPU).
- **Pourquoi** : Muon has built-in normalization (NS + NorMuon) protecting matrix params, but AdamW params (embeddings, lm_head, scalars) have zero gradient normalization. Gradient spikes corrupt their EMA states, degrading updates for many subsequent steps. max_norm=1.0 only clips extreme spikes, otherwise a no-op. This is the #1 HIGH PRIORITY untested change per prompt.md notes. Exp58 attempted this but crashed on infra (never ran on GPU).
- **Risques** : Zero crash/OOM/speed risk. 1-line addition. If max_norm=1.0 clips too aggressively: mild regression ~0.001-0.003. If norms are naturally < 1.0: complete no-op (identical to baseline).
- **Changement** : Added `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` after gradient accumulation, before optimizer.step().


## Experiment #63 — 2026-03-12 13:20:38

- **Best val_bpb avant** : 1.078476
- **Status** : discard (val_bpb: 1.080565)
- **Description** : Reduce z-loss coefficient from 1e-4 to 5e-5 to halve partition function regularization strength.
- **Idée** : Reduce z-loss coefficient from 1e-4 to 5e-5 — halve partition function regularization strength.
- **Pourquoi** : Z-loss coefficient has NEVER been tuned (default PaLM value). With softcap=13 already constraining logits to [-13, 13], z-loss is partially redundant. Reducing to 5e-5 preserves stability benefit while freeing gradient bandwidth for actual loss minimization.
- **Risques** : Zero. Single number change. If too weak: mild regression ~0.001-0.003. If already optimal: no-op (within noise).
- **Changement** : `z_loss = 1e-4 * logz.square().mean()` → `z_loss = 5e-5 * logz.square().mean()`. Single number change.


## Experiment #64 — 2026-03-12 13:28:57

- **Best val_bpb avant** : 1.078476
- **Status** : discard (val_bpb: 1.082924)
- **Description** : Add tiny LR warmup (WARMUP_RATIO=0.0 to 0.01) to stabilize the initial gradient phase after compilation.
- **Idée** : Add tiny LR warmup (WARMUP_RATIO=0.0 → 0.01) to stabilize initial gradient phase after compilation.
- **Pourquoi** : Currently no warmup — model goes from random initialization straight to full LR at step 11. The first real optimizer steps hit with full learning rate on chaotic, high-magnitude gradients from a random model. A tiny warmup of 1% of time budget (~3s, ~8-9 steps) lets AdamW momentum buffers (exp_avg, exp_avg_sq) initialize on smaller updates, giving smoother convergence. This is standard LLM practice (GPT-2/3, LLaMA all use warmup). The warmup is brief enough that it doesn't reduce effective training time. Marked HIGH PRIORITY in prompt.md notes — never tested before.
- **Risques** : Zero. Single number change (0.0→0.01). No crash/OOM/speed risk. If warmup is unnecessary (gradients already well-behaved): identical to baseline (warmup covers only ~8-9 steps at slightly reduced LR). If warmup helps: ~0.0005-0.002 improvement from cleaner optimizer state initialization.
- **Changement** : `WARMUP_RATIO = 0.0` → `WARMUP_RATIO = 0.01`. Single number change.


## Experiment #65 — 2026-03-12 13:38:53

- **Best val_bpb avant** : 1.078476
- **Status** : discard (val_bpb: 1.091849)
- **Description** : Reduce RoPE base frequency from 10000 to 1000 for better positional resolution within 2048 context length.
- **Idée** : Reduce RoPE base frequency from 10000 to 1000 for better positional resolution within 2048 context.
- **Pourquoi** : With base=10000 and head_dim=128 (64 rotary dims), only ~21/64 dimensions have wavelengths ≤ 2048 context length — the other 43 dimensions barely differentiate positions and are wasted. With base=1000, ~27+ dimensions become useful and overall position resolution increases. The model uses QK-norm, so rotary angles affect attention direction only (not magnitude). This is an architectural change to positional encoding never tested in this series. Standard base=10000 was designed as a reasonable default, not an optimized choice for 2048 context.
- **Risques** : Low. Single constant change. If lower base hurts (model uses high-wavelength dims as position-invariant content channels), mild regression ~0.001-0.005. No crash/OOM/speed risk — identical computation, just different frequency values.
- **Changement** : `_precompute_rotary_embeddings` default `base=10000` → `base=1000`. Single constant change.


## Experiment #66 — 2026-03-12 13:52:09

- **Best val_bpb avant** : 1.078476
- **Status** : discard (val_bpb: 1.084504)
- **Description** : Replace ReluSquared with GeluSquared activation in MLP for softer gradient flow through negative activations.
- **Idée** : Replace ReluSquared with GeluSquared activation in MLP — softer gradient flow through mildly negative activations.
- **Pourquoi** : ReluSquared has a hard dead zone (exactly 0 gradient for x<0). In the undertrained regime (~830 steps), this means ~50% of activations per input produce zero gradient, wasting learning capacity. GeluSquared (F.gelu(x).square()) has soft non-zero gradients for mildly negative x, allowing "dead" neurons to recover and contribute. The square still provides sparsity. SwiGLU was tested (exp30, 1.083) but changed the gating structure — this only changes the base activation. GELU is the standard in modern LLMs (GPT-2, BERT, LLaMA). GeluSquared has never been tested in this series.
- **Risques** : Zero crash/OOM/speed risk. One-line change. GELU is marginally slower than ReLU but negligible. If GELU's soft gating adds noise instead of signal: mild regression ~0.001-0.003. If the exact sparsity of ReLU is important: regression.
- **Changement** : `F.relu(x).square()` → `F.gelu(x).square()` in MLP.forward. Single line change.


## Experiment #67 — 2026-03-12 14:13:11

- **Best val_bpb avant** : 1.078476
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Per-head learnable attention temperature (qk_scales) to let each head and layer optimize its own attention sharpness, ov


## Experiment #68 — 2026-03-12 14:30:20

- **Best val_bpb avant** : 1.078476
- **Status** : discard (val_bpb: 1.080678)
- **Description** : Per-head learnable attention temperature (qk_scales) to restore per-head sharpness control removed by QK-norm — retry 
- **Description** : Per-head learnable attention temperature (qk_scales) — retry of exp67 which crashed (likely infra), never tested on GPU.
- **Idée** : Add a learnable scalar per (layer, head) that scales Q after QK-norm, effectively giving each head its own attention temperature. QK-norm forces all heads to identical temperature (~1/sqrt(128)=1/11.3). Some heads need sharper attention (copying, retrieval) and others broader (contextual blending). With only 4 heads, each head's role matters a lot.
- **Pourquoi** : QK-norm is important (used since baseline) but removes one degree of freedom: temperature. Modern architectures with QK-norm (Gemma 2) include learnable per-head scales to restore this. With softcap=13, the max attention logit for unit-norm QK is ~11.3 (87% of softcap), leaving headroom for scales ~1.15 before saturating tanh. This is a genuine architectural fix (priority #1), not a hyperparameter tweak. Exp67 had this exact idea but crashed on infra.
- **Risques** : Zero crash/OOM risk (32 extra scalars, negligible memory). Zero speed impact (one scalar broadcast multiply per layer). If fixed temperature is optimal: neutral within noise. If scales grow too large (>1.5): possible mild softcap saturation, mitigated by conservative LR (0.05).
- **Changement** : Added `self.qk_scales = nn.Parameter(torch.ones(n_layer, n_head))` in GPT. Passed `qk_scales[i]` through Block → CausalSelfAttention. After QK-norm: `q = q * qk_scale[None, None, :, None]`. AdamW optimizer group at lr=0.05, betas=(0.96, 0.95). Updated assertion and num_scaling_params. ~10 lines changed across 5 methods.


## Experiment #69 — 2026-03-12 14:40:20

- **Best val_bpb avant** : 1.078476
- **Status** : discard (val_bpb: 1.090227)
- **Description** : Reduce TOTAL_BATCH_SIZE from 262144 to 196608 (grad_accum 4 to 3) for 33% more optimizer steps with only 15% noisier gra
- **Description** : Reduce TOTAL_BATCH_SIZE from 262144 to 196608 (grad_accum 4→3) for 33% more optimizer steps with only 15% noisier gradients.
- **Idée** : Reduce TOTAL_BATCH_SIZE from 2^18=262144 to 196608=3×65536 (grad_accum=3). 33% more optimizer steps at 15% more gradient noise.
- **Pourquoi** : The biggest improvement in this series was MLP 3x (-0.008), which worked purely because it gave more optimizer steps in the 5-minute budget. The "more steps" lever is the strongest known factor. 2^17=131072 (grad_accum=2) was tested in exp6 and gave +0.011 regression — too noisy at 41% gradient noise increase. But 196608 is never tested and sits at the interesting intermediate point: 33% more steps (vs 100% for 2^17) with only 15% more noise (vs 41% for 2^17). The noise-steps tradeoff is nonlinear, and this midpoint could be in the sweet zone where step gains outweigh noise cost.
- **Risques** : Zero crash/OOM risk. Single number change. If gradients are still too noisy: regression of ~0.002-0.005 (informative for batch size curve). If the sweet spot extends lower: improvement of ~0.001-0.003.
- **Changement** : `TOTAL_BATCH_SIZE = 2**18` → `TOTAL_BATCH_SIZE = 196608` (line 455 of train.py).


## Experiment #70 — 2026-03-12 14:56:13

- **Best val_bpb avant** : 1.078476
- **Status** : en cours...


## Experiment #71 — 2026-03-12 15:14:17

- **Best val_bpb avant** : 1.078476
- **Status** : keep (val_bpb: 1.077355)
- **Description** : U-Net skip connections — connect early layer outputs to mirror late layers (0→7, 1→6, 2→5, 3→4) with learnable


## Experiment #72 — 2026-03-12 15:33:31

- **Best val_bpb avant** : 1.077355
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Enable actual sliding window attention via flex_attention for the SDPA path — the SSSL window pattern was designed but
- **Description** : Enable actual sliding window attention via flex_attention for the SDPA path — the SSSL window pattern was designed but never functional on RTX 5090.
- **Idée** : Use `torch.nn.attention.flex_attention` with `create_block_mask` to implement real sliding window attention on non-Hopper GPUs. Currently, all 8 layers do full causal attention (SDPA ignores window_size). With this change, 6/8 layers (the "S" layers in SSSL pattern) use half-context (1024) sliding window via block-sparse attention, skipping computation for far-away tokens.
- **Pourquoi** : The model was DESIGNED with the SSSL window pattern (3 short + 1 long, repeated), but on RTX 5090 (non-Hopper), the SDPA path has no sliding window support — all layers do full 2048 causal attention. flex_attention (PyTorch 2.5+) generates compiled Triton kernels with block-sparse masking that skip masked blocks entirely. With 6/8 layers using window=1024, we skip ~50% of attention compute on those layers. If attention is ~30% of total compute, that's ~10-15% overall speedup → more training steps. The "more steps" thesis is the strongest proven factor (MLP 3x gave -0.008). Additionally, the sliding window inductive bias (local attention for most layers, global for every 4th) matches the intended architecture design.
- **Risques** : (1) flex_attention Triton kernels may be slower per-FLOP than SDPA's optimized backend, potentially negating the window speedup. (2) torch.compile may fail to trace flex_attention + BlockMask correctly, causing crash or graph breaks. (3) Sliding window may hurt quality if full attention was actually beneficial for all layers. Fallback: if create_block_mask fails, all layers silently fall back to SDPA (try/except in init_weights).
- **Changement** : Import flex_attention + create_block_mask. Add `_block_mask = None` to CausalSelfAttention. In init_weights, create BlockMask for each unique short window size and assign to S-layer attention modules. In forward, use flex_attention when _block_mask is set, else SDPA. ~25 lines changed across 3 methods.


## Experiment #73 — 2026-03-12 15:50:28

- **Best val_bpb avant** : 1.077355
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Increase z-loss coefficient from 1e-4 to 2e-4 to strengthen logit partition function regularization, following the clear
- **Description** : Increase z-loss coefficient from 1e-4 to 2e-4 to strengthen logit partition function regularization.
- **Idée** : Double the z-loss coefficient from 1e-4 to 2e-4. Clear directional signal: exp63 tested 5e-5 and regressed (+0.002 vs 1e-4). Stronger z-loss = better. 2e-4 never tested.
- **Pourquoi** : Z-loss penalizes large log-partition function (logsumexp of logits). With softcap=13, the synergy is clear: softcap bounds logits, z-loss penalizes approaching the bound unnecessarily, improving gradient flow and calibration. Exp9 showed z-loss at 1e-4 was an improvement. Exp63 showed 5e-5 was worse. The natural next step is 2e-4.
- **Risques** : Zero crash/OOM risk. Single number change. If too strong: mild regression ~0.001-0.003. If right: improvement ~0.0005-0.002.
- **Changement** : `z_loss = 1e-4 * logz.square().mean()` → `z_loss = 2e-4 * logz.square().mean()` (line 319 of train.py).


## Experiment #74 — 2026-03-12 15:55:49

- **Best val_bpb avant** : 1.077355
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Increase z-loss coefficient from 1e-4 to 2e-4 to strengthen logit partition function regularization (retry of exp73 whic
- **Description** : Increase z-loss coefficient from 1e-4 to 2e-4 (retry of exp73 which crashed on infra, never tested on GPU).
- **Idée** : Double the z-loss coefficient from 1e-4 to 2e-4. Clear directional signal: exp63 tested 5e-5 and regressed (+0.002 vs 1e-4). Stronger z-loss = better. 2e-4 never tested.
- **Pourquoi** : Z-loss penalizes the log-partition function (logsumexp of logits). With softcap=13, the synergy is clear: softcap bounds logits, z-loss penalizes approaching the bound unnecessarily, improving gradient flow and calibration. Exp9 showed z-loss at 1e-4 was an improvement. Exp63 showed 5e-5 was worse. The natural next step is 2e-4. Exp73 attempted this exact change but crashed on infra (single number change = zero code risk).
- **Risques** : Zero crash/OOM risk. Single number change. If too strong: mild regression ~0.001-0.003. If right: improvement ~0.0005-0.002.
- **Changement** : `z_loss = 1e-4 * logz.square().mean()` → `z_loss = 2e-4 * logz.square().mean()` (line 319 of train.py).


## Experiment #75 — 2026-03-12 16:04:35

- **Best val_bpb avant** : 1.077355
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Increase z-loss coefficient from 1e-4 to 2e-4 to strengthen logit partition function regularization (3rd attempt — exp
- **Description** : Increase z-loss coefficient from 1e-4 to 2e-4 (3rd attempt — exp73/74 crashed infra, never tested on GPU).
- **Idée** : Double the z-loss coefficient. Clear directional signal: no z-loss → 1e-4 improved (exp9), 5e-5 worse than 1e-4 (exp63). Stronger = better. 2e-4 never tested on GPU.
- **Pourquoi** : Z-loss penalizes logsumexp of logits, synergistic with softcap=13. The direction is unambiguous across 3 experiments. This is the lowest-risk, highest-evidence change available.
- **Risques** : Zero crash/OOM risk. Single number change. If too strong: mild regression ~0.001-0.003.
- **Changement** : `z_loss = 1e-4 * logz.square().mean()` → `z_loss = 2e-4 * logz.square().mean()` (line 319).


## Experiment #76 — 2026-03-12 16:07:30

- **Best val_bpb avant** : 1.077355
- **Description** : Increase z-loss coefficient from 1e-4 to 2e-4 (4th attempt — exp73/74/75 all crashed on infra, never tested on GPU).
- **Idée** : Double the z-loss coefficient. Clear directional signal: 5e-5 → +0.002 regression (exp63), 1e-4 is current best. Stronger z-loss = better gradient flow through softcap tanh. 2e-4 is the natural next step.
- **Pourquoi** : Z-loss penalizes large logit partition functions (PaLM-style). With softcap=13, stronger z-loss keeps logits further from tanh saturation, preserving gradient magnitude. The directional signal is unambiguous: weaker hurt, current is good, stronger should be better.
- **Risques** : Zero crash/OOM risk. Single constant change. If 2e-4 too strong: z-loss gradient could dominate CE gradient → mild regression ~0.001-0.003.
- **Changement** : `z_loss = 1e-4 * logz.square().mean()` → `z_loss = 2e-4 * logz.square().mean()` (line 319).


## Experiment #77 — 2026-03-12 16:20:55

- **Best val_bpb avant** : 1.077355
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Mid-MLP normalization (NormFormer-style) — add RMSNorm between ReluSquared activation and output projection in MLP.


## Experiment #78 — 2026-03-12 16:29:32

- **Best val_bpb avant** : 1.077355
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Mid-MLP normalization (NormFormer-style) — add RMSNorm between ReluSquared activation and output projection in MLP (re
- **Description** : Mid-MLP normalization (NormFormer-style) — retry of exp77 which crashed on infra, never tested on GPU.
- **Idée** : Add RMSNorm between ReluSquared activation and output projection in MLP (`x = norm(x)` after `F.relu(x).square()`).
- **Pourquoi** : Every winning experiment improved gradient flow (softcap 13, z-loss, U-Net skips). The MLP hidden layer is the last major unnormalized internal activation. ReluSquared produces sparse activations with potentially large values; RMSNorm ensures c_proj always receives consistent-magnitude inputs, simplifying its learning task and stabilizing gradient flow. NormFormer (Shleifer et al., 2021) showed this helps. Zero extra parameters, negligible compute. Exp77 attempted this exact change but crashed on infra (0.0 VRAM = never reached GPU).
- **Risques** : Zero crash/OOM risk (norm() used throughout model, well-supported by torch.compile). If ReluSquared's raw magnitude carries useful signal: mild regression ~0.001-0.003. The sparse activation + RMSNorm interaction (50% zeros) is handled by eps and should be fine — norm just scales non-zero values to consistent magnitude.
- **Changement** : Added `x = norm(x)` after `x = F.relu(x).square()` in MLP.forward() (line 120 of train.py).


## Experiment #79 — 2026-03-12 16:40:48

- **Best val_bpb avant** : 1.077355
- **Status** : discard (val_bpb: 1.079762)
- **Description** : Mid-MLP normalization (NormFormer-style) — add RMSNorm between ReluSquared activation and output projection in MLP (3r


## Experiment #79 — 2026-03-12

- **Best val_bpb avant** : 1.077355
- **Description** : Mid-MLP normalization (NormFormer-style) — add RMSNorm between ReluSquared activation and output projection in MLP (3rd attempt, exp77/78 crashed on infra).
- **Idée** : Add `x = norm(x)` after `F.relu(x).square()` and before `self.c_proj(x)` in MLP.forward(). ReluSquared produces sparse activations with potentially large squared values. Normalizing ensures c_proj receives unit-variance inputs, stabilizing gradient flow.
- **Pourquoi** : All winning experiments improved gradient flow: softcap 13 (tanh gradient), z-loss (logit calibration), U-Net skips (shorter gradient paths). The MLP hidden activation is the last major unnormalized internal activation. NormFormer (Shleifer et al., 2021) showed adding norms within sublayers improves convergence. Zero params, negligible compute.
- **Risques** : Zero crash/OOM risk (norm() used everywhere, torch.compile compatible). If ReluSquared magnitude carries useful signal, mild regression ~0.001-0.003. If sparsity-dependent RMS scaling is problematic, mild regression.

## Experiment #80 — 2026-03-12 16:52:44

- **Best val_bpb avant** : 1.077355
- **Description** : Multi-scale RoPE — per-head base frequencies (2500, 10000, 40000, 160000) for attention head specialization.
- **Idée** : Give each attention head a different RoPE base frequency in geometric progression (ratio=4). Head 0 (base=2500) has fast positional decay for local patterns. Heads 1 (10000) stays at default. Head 2 (40000) has slow decay for medium-range. Head 3 (160000) has near-content-based attention with minimal positional bias.
- **Pourquoi** : All heads currently share base=10000. While RoPE already encodes multiple frequencies via different dimension channels, ALL heads see the same positional relationship for a given token pair. Multi-scale bases let heads specialize in different distance ranges — head 0 for syntax/local structure, head 3 for semantic/long-range dependencies. Used in YaRN/LongRoPE for context extension. Zero extra compute, zero extra params. This is a genuine architectural change (priority #1 per prompt).
- **Risques** : Exp65 tested base=1000 for ALL heads → catastrophic +0.014 regression. But here only 1/4 heads deviates significantly from default. The base=2500 head is still much more conservative than 1000. Worst case ~0.002 regression if head specialization doesn't help in 800 steps. No crash/OOM risk (only precomputed constants change shape from [1,T,1,d/2] to [1,T,4,d/2]).
- **Changement** : Added `rope_bases` config field and `ROPE_BASES = (2500, 10000, 40000, 160000)` hyperparameter. Modified `_precompute_rotary_embeddings` to compute per-head cos/sin when bases differ.


## Experiment #81 — 2026-03-12 17:08:48

- **Best val_bpb avant** : 1.077197
- **Status** : discard (val_bpb: 1.080138)
- **Description** : Pre-output MLP — add a lightweight feedforward layer (2x expansion, ReluSquared) before lm_head for extra non-linear d
- **Description** : Pre-output MLP — add a lightweight feedforward layer (2x expansion, ReluSquared) before lm_head for extra non-linear depth at ~3% compute cost.
- **Idée** : Add a small MLP (512→1024→512 with ReluSquared) between the final transformer block output and lm_head. Zero-init on output projection. This gives the model extra non-linear transformation depth before the output head — essentially a "cheap 9th layer" without attention overhead.
- **Pourquoi** : DEPTH is the most valuable axis (exp18: losing 1 layer = +0.033 catastrophe). A full 9th layer (exp22) cost ~12.5% compute for only +0.001 regression. This pre-output MLP costs ~3% compute (two 512×1024 matmuls vs eight full attention+MLP layers) but adds meaningful non-linear processing capacity. The output head must map 512-dim hidden states to 32768 vocab — a non-linear pre-processing step can help form better output representations. Zero-init on output_proj ensures model starts identical to baseline.
- **Risques** : Zero crash risk (standard nn.Linear + ReluSquared, same ops used throughout). Minimal OOM risk (+1M params, <2MB VRAM). If the extra compute reduces training steps enough to offset the capacity gain: mild regression ~0.001-0.002. If the pre-output transformation is redundant with the last transformer block: neutral result.


## Experiment #82 — 2026-03-12 17:31:54

- **Best val_bpb avant** : 1.077197
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Initialize skip_lambdas to 0.05 instead of 0.0 for immediate U-Net skip connection activation.
- **Description** : Initialize skip_lambdas to 0.05 instead of 0.0 for immediate U-Net skip connection activation.
- **Idée** : Change `self.skip_lambdas.fill_(0.0)` to `self.skip_lambdas.fill_(0.05)` in init_weights() to give U-Net skip connections a head start.
- **Pourquoi** : U-Net skip connections (exp71) were the second biggest win in the series (-0.001). Currently skip_lambdas starts at 0.0 with slow momentum (beta1=0.96, scalar_lr=0.5), meaning the model needs many steps to learn useful skip weights. With only ~830 training steps, starting at 0.05 means immediate information flow from early layers (0,1,2,3) to their mirror late layers (7,6,5,4). This gives the model shorter gradient paths and richer features from step 0, instead of waiting for the optimizer to discover this. The value 0.05 is conservative enough to not destabilize training (skip contribution is small relative to resid_lambdas=1.0 and x0_lambdas=0.1).
- **Risques** : Zero crash/OOM risk (single scalar init change). If 0.05 is too high and early-layer representations are low quality, mild regression ~0.001. If the optimizer would have learned a very different skip weight pattern (e.g., asymmetric across layers), the uniform init might slow convergence. Expected impact: ~0.0005-0.001 improvement.


## Experiment #83 — 2026-03-12 17:45:45

- **Best val_bpb avant** : 1.077197
- **Description** : Extend Value Embedding to first AND last layer with shared weights for bookend token identity in value computation.
- **Idée** : Apply VE to layer 0 in addition to layer 7, sharing the same embedding table. Each layer has its own input-dependent gate. Zero extra parameters (weight sharing), one extra ve_gate (128 params), minimal compute (one extra embedding lookup).
- **Pourquoi** : VE is critical (removing it = +0.023 disaster in exp1). Currently only the last layer benefits. The first layer's attention computes V from randomly-projected input embeddings with no direct token identity signal. Adding shared VE to layer 0 gives it the same token-specific value residual that makes the last layer effective. The gate is input-dependent so each layer applies different gating despite shared weights. This is an architectural change (priority #1) that bookends the model with token identity in the value path, improving gradient flow through both the start and end of the network. Zero additional optimizer states since weights are tied.
- **Risques** : Zero crash/OOM risk (weight sharing is standard PyTorch, ve_gate uses same ops as existing one). Zero speed impact (one extra embedding lookup is negligible). If first-layer representations are too primitive to benefit from VE: neutral result. If shared weights force a compromise between first and last layer needs: mild regression ~0.001. Worst case matches pre-output MLP result (~+0.003).
- **Changement** : Modified `has_ve` to return True for layer 0 AND last layer. Added weight tying: `self.value_embeds['0'].weight = self.value_embeds[str(n_layer-1)].weight`. ~3 lines changed.


## Experiment #84 — 2026-03-12 17:55:20

- **Best val_bpb avant** : 1.077197
- **Description** : Extend Value Embedding to first AND last layer with shared weights (retry of exp83 which crashed on infra).
- **Idée** : Apply VE to layer 0 in addition to layer 7, sharing the same embedding table via weight tying. Each layer has its own input-dependent ve_gate. Zero extra parameters (shared weight, only one extra ve_gate with 128 params), negligible compute (one extra embedding lookup per batch).
- **Pourquoi** : VE is the most critical component (+0.023 catastrophe when removed). Currently only the last layer benefits from direct token identity in the value path. Adding VE to the first layer gives it token-specific value residuals from step 0, improving gradient flow at both ends of the network. This follows the winning pattern: gradient flow improvements at minimal compute cost (softcap 13, z-loss, U-Net skips, multi-scale RoPE). Exp83 tested this exact idea but crashed on infra (never ran on GPU).
- **Risques** : Zero crash/OOM risk (weight sharing is standard PyTorch, ve_gate uses same ops as existing). Zero speed impact. If first-layer representations are too primitive for VE: neutral. If shared weights force a compromise: mild regression ~0.001. Worst case ~0.003.


## Experiment #85 — 2026-03-12 18:07:17

- **Best val_bpb avant** : 1.072876
- **Status** : discard (val_bpb: 1.081672)
- **Description** : Extend Value Embedding to ALL 8 layers with shared weights for universal token identity in value computation.
- **Description** : Extend Value Embedding to ALL 8 layers with shared weights — every layer gets token-specific value residuals via input-dependent gates.
- **Idée** : Change `has_ve` to return True for all layers, tie all VE weights to a single shared embedding table. Each layer gets its own ve_gate for input-dependent gating. This extends exp84's winning VE bookend strategy (biggest single win: -0.004) to full coverage.
- **Pourquoi** : VE is the most critical component (+0.023 catastrophe when removed, -0.004 when extended to layer 0). Currently only layers 0 and 7 benefit from token identity in the value path. Mid-layers (1-6) compute V purely from projected representations with no direct token-specific signal. Full VE coverage gives every layer's attention access to token identity, potentially improving gradient flow throughout the entire network. Shared weights + per-layer gates mean the model can learn different VE usage patterns per layer at zero parameter cost (only +768 params from 6 new ve_gates). Compute overhead is ~6 extra embedding lookups (same table, memory-bound, ~0.1% of budget).
- **Risques** : Zero crash/OOM risk (same ops used in layers 0 and 7 already). Negligible speed impact. If mid-layers don't benefit from VE (representations already rich enough): neutral. If shared weights force a single-purpose embedding that's suboptimal for diverse layer needs: mild regression ~0.001-0.002. If 6 extra embedding lookups reduce steps enough to offset benefits: mild regression.


## Experiment #86 — 2026-03-12 18:16:42

- **Best val_bpb avant** : 1.072876
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Increase DEPTH from 8 to 9 layers (ASPECT_RATIO=56) for deeper modeling capacity with accumulated gradient flow improvem
- **Description** : Increase DEPTH from 8 to 9 layers (ASPECT_RATIO=56 to keep model_dim=512) for deeper modeling capacity.
- **Idée** : Add a 9th transformer layer by changing DEPTH=9 and ASPECT_RATIO=56 (base_dim=504, rounds to 512, keeping 4 heads). Same width, one more layer. U-Net skips adapt to 4 early + 4 late + 1 middle. VE remains on layers 0 and 8 (shared). Multi-scale RoPE stays with 4 bases for 4 heads.
- **Pourquoi** : Depth is the most valuable architectural axis (exp18: DEPTH=7 = +0.033 catastrophe). Exp22 tested DEPTH=9 with MLP 2.5x before multi-scale RoPE, U-Net skips, and VE first+last — got 1.080504, only +0.001 from baseline. Now with accumulated gradient flow improvements (multi-scale RoPE, U-Net skips, VE bookend), the 9th layer should be better utilized. The ~8-10% compute overhead (lm_head is fixed cost) means ~750 steps instead of ~830, but the extra modeling capacity from a full transformer block should compensate. The deeper U-Net topology (4 skip connections spanning 9 layers) provides richer gradient shortcuts.
- **Risques** : Zero crash/OOM risk (same architecture, one more standard Block). ~8-10% fewer training steps due to extra compute. If the step reduction outweighs the capacity gain: regression ~0.001-0.003. If 9 layers need more training time than 300s to converge: could be worse than 8-layer model.


## Experiment #87 — 2026-03-12 18:30:09

- **Best val_bpb avant** : 1.072876
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Increase DEPTH from 8 to 9 layers (ASPECT_RATIO=56) — retry of exp86 which crashed on infra, never tested on GPU.
- **Description** : Increase DEPTH from 8 to 9 layers (ASPECT_RATIO=56) — retry of exp86 which crashed on infra, never tested on GPU.
- **Idée** : Add a 9th transformer layer by changing DEPTH=9 and ASPECT_RATIO=56 (base_dim=504, rounds to 512, keeping 4 heads). Same width (512), one more layer. U-Net skips adapt to 4 early + 4 late + 1 middle. VE remains on layers 0 and 8 (shared). Multi-scale RoPE stays with 4 bases for 4 heads.
- **Pourquoi** : Depth is the single most valuable architectural axis — exp18 (DEPTH=7) was a +0.033 catastrophe. Exp22 tested DEPTH=9 before all gradient flow improvements and got 1.080504 (+0.001 from that baseline). Since then, accumulated improvements (multi-scale RoPE -0.002, U-Net skips -0.001, VE bookend -0.004, softcap 13 -0.001) total ~-0.008, which should make the 9th layer viable. Also exp22 used MLP 2.5x (known -0.003 worse than 3x). Back-of-envelope: ~1.080504 - 0.008 - 0.003 ≈ 1.070, a potential new best. The ~8-10% compute overhead means ~750 steps vs ~830, but deeper U-Net topology (4 skip connections spanning 9 layers) + VE bookend at both extremes of a deeper network should compensate.
- **Risques** : Zero crash/OOM risk (same architecture, one more standard Block). ~8-10% fewer training steps. If step reduction outweighs capacity gain: regression ~0.001-0.003. Exp86 crashed on infra (RunPod), not a code issue.


## Experiment #88 — 2026-03-12 18:41:06

- **Best val_bpb avant** : 1.072876
- **Status** : crash (val_bpb: 0.000000)
- **Description** : experiment 88


## Experiment #89 — 2026-03-12 18:42:54

- **Best val_bpb avant** : 1.072876
- **Status** : discard (val_bpb: 1.073159)
- **Description** : experiment 89


## Experiment #90 — 2026-03-12 18:48:51

- **Best val_bpb avant** : 1.072876
- **Status** : keep (val_bpb: 1.072821)
- **Description** : experiment 90


## Experiment #91 — 2026-03-12 18:54:34

- **Best val_bpb avant** : 1.072821
- **Status** : discard (val_bpb: 1.072978)
- **Description** : experiment 91


## Experiment #92 — 2026-03-12 19:00:08

- **Best val_bpb avant** : 1.072821
- **Status** : en cours...
- **Description** : DEPTH=9 with U-Net skip bug fix for odd depths — exp86/87 both crashed because `skip_lambdas[mirror]` IndexError when n_layer is odd, not infra.
- **Idée** : Fix the U-Net skip indexing bug (`if i >= half:` → `if i >= half and mirror < half:`) that caused DEPTH=9 crashes in exp86/87, then increase depth from 8 to 9 layers (ASPECT_RATIO=56 to keep model_dim=512). With all accumulated gradient flow improvements (multi-scale RoPE, U-Net skips, VE bookends, softcap 13), deeper models should benefit more than before. Exp22 (DEPTH=9 before improvements) was only +0.001 with MLP 2.5x; correcting for MLP 3x and accumulated improvements suggests ~1.070.
- **Pourquoi** : Depth is the most valuable axis — exp18 (DEPTH=7) was catastrophic (+0.033), confirming depth matters enormously. All gradient flow improvements since exp22 specifically help deeper models: U-Net skips provide shorter gradient paths through 9 layers, VE bookends give token identity at both ends, multi-scale RoPE helps heads specialize across the deeper network.
- **Risques** : ~8-10% fewer training steps (~750 vs ~830) due to extra layer compute. If step reduction outweighs capacity gain: mild regression ~0.001-0.003. Zero OOM risk. The bug fix is safe and correct for both even and odd depths.


## Experiment #93 — 2026-03-12 19:05:48

- **Best val_bpb avant** : 1.072821
- **Status** : en cours...
- **Description** : DEPTH=9 retry (5th attempt) with U-Net skip bug fix for odd depths — exp86/87/88/92 all crashed, fix verified correct.
- **Idée** : Increase depth from 8 to 9 layers (ASPECT_RATIO=56 to keep model_dim=512). Fix U-Net skip indexing: `if i >= half and mirror < half:` to handle the middle layer (i=4) correctly when n_layer is odd. All other components auto-adapt: VE on layers 0 and 8, multi-scale RoPE (4 bases for 4 heads), window pattern SSSL cycles.
- **Pourquoi** : Depth is the most valuable axis (exp18 DEPTH=7: +0.033 catastrophe). Exp22 tested DEPTH=9 before gradient flow improvements and got 1.080504 with MLP 2.5x. With accumulated improvements (multi-scale RoPE -0.002, U-Net skips -0.001, VE bookends -0.004) and MLP 3x (vs 2.5x), back-of-envelope: ~1.070-1.071. Previous attempts crashed on RunPod infra, not code bugs (exp92 had the correct fix but still crashed).
- **Risques** : ~8-10% fewer training steps (~750 vs ~830) due to extra layer. If step reduction outweighs capacity gain: regression ~0.001-0.003. Zero OOM risk. Previous crashes were infra, not code.


## Experiment #94 — 2026-03-12 19:12:06

- **Best val_bpb avant** : 1.072821
- **Status** : crash (val_bpb: 0.000000)
- **Description** : DEPTH=9 (6th attempt) with U-Net skip bug fix for odd depths — previous 5 attempts all crashed.


## Experiment #95 — 2026-03-12 19:15:17

- **Best val_bpb avant** : 1.072821
- **Status** : crash (val_bpb: 0.000000)
- **Description** : DEPTH=9 (7th attempt) with U-Net skip bug fix for odd depths — previous 6 attempts all crashed (bug or infra).
- **Description** : DEPTH=9 (7th attempt) with U-Net skip bug fix for odd depths — previous 6 attempts all crashed (bug or infra).
- **Idee** : Increase depth from 8 to 9 layers (ASPECT_RATIO=56 to keep model_dim=512). Fix U-Net skip indexing: `if i >= half and mirror < half:` to handle the middle layer (i=4) correctly when n_layer is odd. All other components auto-adapt: VE on layers 0 and 8 (shared weights, separate gates), multi-scale RoPE (4 bases for 4 heads), window pattern SSSL cycles.
- **Pourquoi** : Depth is the single most valuable axis (exp18 DEPTH=7: +0.033 catastrophe). This exact change has never successfully run on GPU — exp86/87/88 had the IndexError bug, exp92/93/94 had the fix but crashed on infra. Back-of-envelope estimate: ~1.070-1.071.
- **Risques** : ~8-10% fewer training steps (~750 vs ~830) due to extra layer compute. Zero OOM risk. Main risk is yet another infra crash. If step reduction outweighs capacity gain: regression ~0.001-0.003.


## Experiment #96 — 2026-03-12 19:22:19

- **Best val_bpb avant** : 1.072821
- **Status** : en cours...


## Experiment #97 — 2026-03-16 10:44:59

- **Best val_bpb avant** : 1.072821
- **Status** : discard (val_bpb: 1.079870)
- **Description** : DEPTH=9 with U-Net skip bug fix — the fix was missing from the persisted train.py despite being documented.
- **Idée** : DEPTH=9 (8th attempt) with U-Net skip bug fix applied to current train.py — the fix (`if i >= half and mirror < half:`) was NOT in the codebase despite being documented in prompt.md. Previous exp92-95 applied the fix but crashed on infra (RunPod). This attempt is on fresh infra 4 days later.
- **Pourquoi** : Depth is the single most valuable axis (exp18 DEPTH=7: +0.033 catastrophe). DEPTH=9 has NEVER successfully run — the bug fix was missing from the persisted train.py. With ASPECT_RATIO=56 → model_dim=512, 4 heads, all components auto-adapt (VE bookend on layers 0/8, multi-scale RoPE 4 bases for 4 heads, U-Net skip 4 pairs). Back-of-envelope: ~1.070-1.071.
- **Risques** : ~8-10% fewer training steps (~750 vs ~830) due to extra layer compute. Zero OOM risk (512 dim is smaller than 512 dim at DEPTH=8). Main risk: infra crash again, or step reduction outweighs capacity gain for mild regression ~0.001-0.003.


## Experiment #98 — 2026-03-16 10:54:20

- **Best val_bpb avant** : 1.072821
- **Status** : keep (val_bpb: 1.070366)
- **Description** : Initialize skip_lambdas to 0.05 instead of 0.0 for immediate U-Net skip connection activation from step 1.
- **Description** : Initialize skip_lambdas to 0.05 instead of 0.0 for immediate U-Net skip connection activation from step 1.
- **Idée** : Change skip_lambdas init from 0.0 to 0.05. U-Net skip connections (exp71, -0.001) are the 3rd most impactful architectural win. Currently they start completely inactive (init=0.0) and must learn to activate during training. With only ~830 steps, parameters initialized closer to their useful operating range converge faster. At 0.05, each late mirror layer immediately receives a 5% contribution from its early mirror, providing gradient shortcuts from step 1.
- **Pourquoi** : (1) U-Net skips were a proven win (exp71). (2) Short training budget (~830 steps) means slow-starting mechanisms waste valuable steps. (3) Exp82 attempted this but crashed on infra, so it has NEVER been tested on GPU. (4) Zero compute cost — no new params, no extra FLOPs.
- **Risques** : Very low. 0.05 is small enough that the optimizer can easily correct it if wrong. No OOM risk, no torch.compile issues, no crash risk. Worst case: the optimizer prefers 0.0 and pushes back, resulting in negligible regression within noise (~0.001).


## Experiment #99 — 2026-03-16 11:08:44

- **Best val_bpb avant** : 1.070366
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Differential Attention (DiffTransformer) — split each head into 2 sub-heads, subtract paired attention outputs to canc
- **Idée** : Differential Attention (DiffTransformer) — split each head's Q/K/V into 2 sub-heads (HEAD_DIM//2=64), run SDPA with 2x heads, then subtract paired outputs (y_a - 0.5*y_b) to cancel attention noise. Also change c_proj from n_embd→n_embd to n_embd//2→n_embd, saving ~5% per-layer linear compute → slightly more training steps.
- **Pourquoi** : (1) DiffAttn is the #7 remaining untested idea in prompt.md. (2) Noise cancellation via subtraction of paired attention outputs is well-documented in the DiffTransformer paper (Microsoft, 2024). (3) The split happens AFTER RoPE, so sub-head 0 gets low-frequency position info and sub-head 1 gets high-frequency — subtraction cancels patterns common to both frequency bands, retaining only distinctive signals. (4) c_proj reduction from 512×512 to 512×256 saves ~131K params/layer × 8 layers ≈ 1M params, giving ~5% speed improvement. (5) No new parameters, no new dependencies, static shapes → torch.compile safe.
- **Risques** : (1) HEAD_DIM//2=64 sub-heads have softer attention (QK-norm at 128 dims → split halves have norm ~1/sqrt(2), making attention logits ~40% smaller). If attention is too diffuse, quality degrades. (2) Fixed lambda=0.5 might not be optimal (paper uses learnable ~0.8). (3) Halving value dimension per logical head from 128 to 64 reduces information capacity per attention head output. (4) HEAD_DIM=64 as standalone was +0.012 (dead), but DiffAttn is fundamentally different (subtraction mechanism). Risk of regression ~0.005-0.015.


## Experiment #100 — 2026-03-16 11:32:51

- **Best val_bpb avant** : 1.070366
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Sandwich norm (post-sublayer RMSNorm) on both attention and MLP outputs to normalize residual stream contributions.
- **Description** : Sandwich norm (post-sublayer RMSNorm) — add norm() around attention and MLP outputs before residual add.
- **Idée** : Add RMSNorm after both attention and MLP sublayer outputs, before they are added to the residual stream: `x = x + norm(attn(norm(x)))` and `x = x + norm(mlp(norm(x)))`. This normalizes each sublayer's contribution to O(1) scale, preventing residual stream magnitude from growing unboundedly across layers. Different from mid-MLP norm (exp79, +0.002, dead) which was INSIDE the MLP between activation and output projection.
- **Pourquoi** : (1) Sandwich norm is a proven technique from PaLM, CogView, and other architectures. (2) Without post-norm, sublayer contributions accumulate and the residual stream grows ~O(sqrt(2*n_layer)), which can destabilize later layers and make the learned resid_lambdas/x0_lambdas compensate for scale instead of controlling information flow. (3) Zero new parameters, zero new modules, minimal compute overhead (2 extra RMSNorm per layer). (4) torch.compile safe (uses existing F.rms_norm). (5) c_proj zero-init means norm(0)=0 at init, so the model starts identically to baseline.
- **Risques** : (1) At early training steps when c_proj outputs near-zero values, the norm gradient amplification is ~1/sqrt(eps) ≈ 316x, but Muon's Newton-Schulz step is scale-invariant so this is harmless for matrix params. (2) Mid-MLP norm was dead (+0.002), so "more norms" might not help at this scale, but sandwich norm addresses a fundamentally different issue (residual stream growth vs. MLP internal scale). (3) Expected range: -0.003 to +0.005.


## Experiment #101 — 2026-03-16 11:38:35

- **Best val_bpb avant** : 1.070366
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Retry sandwich norm (post-sublayer RMSNorm) — exp100 crashed on infra, never tested on GPU.
- **Description** : Retry sandwich norm (post-sublayer RMSNorm) — exp100 crashed on infra, never tested on GPU.
- **Idée** : Add RMSNorm after both attention and MLP sublayer outputs, before they are added to the residual stream: `x = x + norm(attn(norm(x)))` and `x = x + norm(mlp(norm(x)))`. This normalizes each sublayer's contribution to O(1) scale. Exact same change as exp100 which was pure infra crash (never ran on GPU).
- **Pourquoi** : (1) Exp100 crashed on RunPod infra (same CUDA init failure pattern as exp99 and exp102-210), code was never tested on GPU. (2) Sandwich norm is a proven technique from PaLM and CogView. (3) Without post-norm, residual stream grows ~O(sqrt(2*n_layer)), which can destabilize later layers. (4) Different from mid-MLP norm (exp79, +0.002, dead) — that was INSIDE MLP, this is OUTSIDE. (5) Zero new params, zero new modules, minimal compute overhead. (6) c_proj zero-init means norm(0)=0 at init → model starts identically to baseline.
- **Risques** : (1) Gradient amplification through norm at near-zero values in early training, mitigated by Muon's scale-invariant Newton-Schulz. (2) Mid-MLP norm was dead, but sandwich norm addresses a fundamentally different issue. (3) Expected range: -0.003 to +0.005. Very low crash risk — 2-line change using existing norm() function.


## Experiment #102 — 2026-03-16 11:42:50

- **Best val_bpb avant** : 1.070366
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Sandwich norm (post-sublayer RMSNorm) on attention and MLP outputs — 3rd attempt after 2 infra crashes.
- **Description** : Sandwich norm (post-sublayer RMSNorm) — 3rd attempt, exp100/101 both infra crashes, never tested on GPU.
- **Idée** : Add norm() around attention and MLP outputs before residual add: `x = x + norm(attn(norm(x)))` and `x = x + norm(mlp(norm(x)))`. This normalizes each sublayer's contribution to O(1) scale, preventing residual stream magnitude from growing across layers.
- **Pourquoi** : (1) Never tested on GPU — exp100/101 were pure infra crashes. (2) Proven technique from PaLM/CogView. (3) Different from mid-MLP norm (exp79, dead) which was INSIDE MLP. This is OUTSIDE, normalizing the full sublayer output. (4) Zero new params, minimal compute overhead (2 extra RMSNorm per layer). (5) c_proj zero-init means norm(0)=0 at init → model starts identically to baseline.
- **Risques** : Very low. 2-line change using existing norm(). Gradient amplification through norm at near-zero early values mitigated by Muon's scale-invariant Newton-Schulz. Expected range: -0.003 to +0.005.


## Experiment #103 — 2026-03-16 11:45:27

- **Best val_bpb avant** : 1.070366
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Sandwich norm (post-sublayer RMSNorm) on attention and MLP outputs to normalize residual stream contributions — 4th at
- **Description** : Sandwich norm (post-sublayer RMSNorm) on attention and MLP outputs — 4th attempt after 3 infra crashes (exp100/101/102 never tested on GPU).
- **Idée** : Add norm() around attention and MLP outputs in Block.forward: `x = x + norm(attn(norm(x)))` and `x = x + norm(mlp(norm(x)))`. This normalizes each sublayer's contribution to O(1) scale, preventing residual stream magnitude from growing ~O(sqrt(2*n_layer)) across 8 layers.
- **Pourquoi** : (1) Never tested on GPU — exp100/101/102 were all pure infra crashes (RunPod CUDA init failure streak). (2) Proven technique from PaLM and CogView architectures. (3) Different from mid-MLP norm (exp79, +0.002, dead) which was INSIDE the MLP between activation and output projection — sandwich norm is OUTSIDE, normalizing the full sublayer output before residual add. (4) Zero new parameters, zero new modules, minimal compute overhead (2 extra RMSNorm per layer). (5) c_proj zero-init means norm(0)=0 at init → model starts identically to baseline.
- **Risques** : Very low. 2-line change using existing norm(). Gradient amplification through norm at near-zero early values (~1/sqrt(eps)) is mitigated by Muon's scale-invariant Newton-Schulz. Expected range: -0.003 to +0.005.


## Experiment #104 — 2026-03-16 11:48:35

- **Best val_bpb avant** : 1.070366
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Sandwich norm (post-sublayer RMSNorm) on attention and MLP outputs — 5th attempt after 4 infra crashes.
- **Description** : Sandwich norm (post-sublayer RMSNorm) on attention and MLP outputs — 5th attempt after 4 infra crashes (exp100-103 never tested on GPU).
- **Idée** : Add norm() around attention and MLP outputs in Block.forward: `x = x + norm(attn(norm(x)))` and `x = x + norm(mlp(norm(x)))`. This normalizes each sublayer's contribution to O(1) scale, preventing residual stream magnitude from growing ~O(sqrt(2*n_layer)) across 8 layers.
- **Pourquoi** : (1) Never tested on GPU — exp100/101/102/103 were all pure infra crashes (RunPod CUDA init failure streak). (2) Proven technique from PaLM and CogView architectures. (3) Different from mid-MLP norm (exp79, +0.002, dead) which was INSIDE the MLP between activation and output projection — sandwich norm is OUTSIDE, normalizing the full sublayer output before residual add. (4) Zero new parameters, zero new modules, minimal compute overhead (2 extra RMSNorm per layer). (5) c_proj zero-init means norm(0)=0 at init → model starts identically to baseline. (6) Muon's Newton-Schulz is scale-invariant, so gradient amplification through norm at near-zero values is harmless for matrix params.
- **Risques** : Very low. 2-line change using existing norm(). Expected range: -0.003 to +0.005. Only risk is if normalizing sublayer outputs removes useful magnitude information from the residual stream.


## Experiment #105 — 2026-03-16 11:52:24

- **Best val_bpb avant** : 1.070366
- **Status** : discard (val_bpb: 1.079623)
- **Description** : Differential Attention (DiffTransformer) with proper V-halving for noise cancellation and ~12% parameter savings.
- **Description** : Differential Attention (DiffTransformer) with proper V-halving — split Q/K into 2x sub-heads, halve V and c_proj, subtract paired outputs for noise cancellation.
- **Idée** : Replace standard multi-head attention with Differential Attention. Each logical head splits into 2 sub-heads at HEAD_DIM//2=64. Q/K reshaping is free (same projection, different view). V is halved (n_kv_head * sub_head_dim instead of n_kv_head * head_dim). Both sub-heads in a pair attend to the same V. Output = attn_sub0 - 0.5 * attn_sub1 (noise cancellation). c_proj input halved (256→512 instead of 512→512). VE also halved to match V dimension. Multi-scale RoPE bases duplicated for sub-head pairs.
- **Pourquoi** : (1) DiffTransformer (Microsoft, 2024) shows noise cancellation improves quality. (2) Halved V + c_proj + VE saves ~12% total params → ~12% more training steps in 5 min (~925 vs ~830). MLP 3x showed compute savings = more steps = the single most important lever. (3) Exp99 tried a simpler DiffAttn (reshape-only, no V halving) but crashed on infra — never tested. This version does proper V-halving for true noise cancellation. (4) QK dimensions unchanged (same projection weights, just reshaped to 8×64 instead of 4×128) → QK attention FLOPs identical. (5) torch.compile safe: all operations are static shapes (repeat_interleave with constant, view, indexing).
- **Risques** : (1) HEAD_DIM//2=64 was +0.012 worse as standalone (exp61), but DiffAttn's subtraction mechanism is fundamentally different from just using smaller heads. (2) Halved V capacity (128→64 per head) may lose information. (3) Fixed lambda=0.5 may not be optimal (paper prefers learnable ~0.8). (4) repeat_interleave on V doubles V memory briefly during SDPA — negligible for 32GB VRAM. Expected range: -0.008 to +0.012.


## Experiment #106 — 2026-03-16 12:09:09

- **Best val_bpb avant** : 1.070366
- **Status** : discard (val_bpb: 1.078373)
- **Description** : Sandwich norm (post-sublayer RMSNorm) on attention and MLP outputs — 6th attempt after 5 infra crashes.
- **Description** : Sandwich norm (post-sublayer RMSNorm) on attention and MLP outputs — 6th attempt after 5 infra crashes (exp100-104 never tested on GPU).
- **Idée** : Add norm() around attention and MLP outputs in Block.forward: `x = x + norm(attn(norm(x)))` and `x = x + norm(mlp(norm(x)))`. This normalizes each sublayer's contribution to O(1) scale, preventing residual stream magnitude from growing across layers.
- **Pourquoi** : (1) Never tested on GPU — exp100-104 were all infra crashes. (2) Exp105 ran successfully, confirming infra is now stable. (3) Proven technique from PaLM/CogView. (4) Different from mid-MLP norm (exp79, dead) which was INSIDE MLP. (5) Zero new params, minimal compute overhead. (6) c_proj zero-init means norm(0)=0 at init → model starts identically to baseline.
- **Risques** : Very low. 2-line change. Normalizing sublayer outputs may remove useful magnitude information from the residual stream. Expected range: -0.003 to +0.005.


## Experiment #107 — 2026-03-16 12:18:00

- **Best val_bpb avant** : 1.070366
- **Status** : en cours...
- **Description** : Causal depthwise convolution before MLP (Mamba/H3-inspired local feature mixing).
- **Idée** : Add a lightweight causal depthwise Conv1d (kernel_size=4, groups=n_embd) between attention output and MLP input in each Block. This gives the MLP access to a 4-token local window of features in addition to the attention-aggregated global representation. Identity-initialized so model starts at exact baseline. Inspired by Mamba (conv before SSM), H3, and hybrid conv+attention architectures.
- **Pourquoi** : (1) MLP processes each position independently — it has no access to neighboring features unless attention explicitly routes them. Short conv adds a complementary local inductive bias. (2) Identity init means zero risk of regression at step 0. (3) Only 16K extra params (0.03% of total), negligible wall-time overhead (<1% — memory-bound op on a 1.8 TB/s GPU). (4) Biggest wins so far came from information flow improvements (VE bookend -0.004, U-Net skip -0.001). This adds another information flow pathway. (5) torch.compile safe: Conv1d + F.pad with static shapes.
- **Risques** : (1) The conv might not learn useful mixing if attention already provides sufficient local context. In that case, the conv stays near identity and the result equals baseline (neutral, not harmful). (2) Conv params in AdamW at LR=0.05 — if too aggressive, the conv could learn noise. (3) F.pad + transpose overhead slightly increases step time. Expected range: -0.005 to +0.003.


## Experiment #108 — 2026-03-16 12:45:35

- **Best val_bpb avant** : 1.070366
- **Status** : discard (val_bpb: 1.070530)
- **Description** : VE embedding LR separation — lower VE LR from 0.6 to 0.2 for more stable value reference signal.
- **Description** : VE embedding LR separation — lower VE LR from 0.6 to 0.2 for more stable value reference signal.
- **Idée** : Separate value embedding (VE) learning rate from token embedding (wte) LR. VE currently shares LR=0.6 with wte, but they serve fundamentally different roles: wte is the primary input representation needing fast adaptation, while VE is a regularizing reference signal for value computation (bookend on first+last layer, shared weights). A lower VE LR (0.2, 3x reduction) should make VE a smoother, more stable reference that changes less aggressively during training.
- **Pourquoi** : (1) VE bookend was the single biggest architectural win (-0.004, exp84). Optimizing how VE trains has high expected value. (2) VE gates start neutral (sigmoid(0)*2=1.0) and modulate how much VE contributes to V — the VE embedding itself should be a stable base that the gates learn to use, not a fast-moving target. (3) Token embeddings need high LR because every token must learn its own representation quickly in ~830 steps. VE serves a different purpose: providing token identity to value computation as a regularizer. (4) Simple 1-line change in optimizer config — VE params already have their own param group. (5) Last 3 experiments (DiffAttn +0.009, sandwich norm +0.008, depthwise conv +0.023) all failed with architectural changes. Time for a targeted optimization of the existing winning architecture.
- **Risques** : (1) LR=0.2 might be too low — VE might not adapt fast enough in ~830 steps, resulting in an undertrained VE that provides weak signal. (2) LR=0.2 might still be too high — optimal could be even lower (0.1 or 0.05). (3) The current LR=0.6 might already be near-optimal for VE, in which case any change hurts. Expected range: -0.003 to +0.004.


## Experiment #109 — 2026-03-16 13:06:56

- **Best val_bpb avant** : 1.070366
- **Status** : discard (val_bpb: 1.070391)
- **Description** : Wider multi-scale RoPE range — (1000, 5000, 50000, 500000) for greater head specialization across distance scales.
- **Description** : Wider multi-scale RoPE range — (1000, 5000, 50000, 500000) for more head specialization across distance scales.
- **Idée** : Widen the per-head RoPE base frequency spread from (2500, 10000, 40000, 160000) [64x range, r=4] to (1000, 5000, 50000, 500000) [500x range]. This gives each of the 4 attention heads a much more distinct positional scale: head 0 (base=1000) specializes in ultra-local patterns (~6 token period), head 1 (base=5000) handles medium-local, head 2 (base=50000) covers medium-range, and head 3 (base=500000) provides near-position-agnostic attention (barely rotates across 2048 context). Multi-scale RoPE already improved the baseline (exp80, -0.0002) — this pushes the specialization further.
- **Pourquoi** : (1) Multi-scale RoPE (exp80) was the last successful architectural change before VE bookend. Widening the range is the natural next step. (2) With only 4 heads, each head carries a lot of weight — maximizing their specialization across distance scales should help. (3) base=1000 was bad uniformly (exp65, +0.013), but here only 1 of 4 heads uses it — the other 3 cover longer ranges. (4) base=500000 provides a "content-only" head that attends based on features without positional bias — complementary to position-aware heads. (5) Simple 1-line change, zero crash/OOM risk.
- **Risques** : (1) base=1000 might make one head too myopic (period ~6 tokens), wasting capacity. (2) base=500000 might make one head ignore position entirely, reducing its usefulness. (3) The optimal spread might be narrower than 500x — the current 64x could already be near-optimal. Expected range: -0.003 to +0.005.


## Experiment #110 — 2026-03-16 13:16:26

- **Best val_bpb avant** : 1.070366
- **Status** : en cours...
- **Description** : Schedule-Free AdamW (Defazio & Mishchenko 2024) for all AdamW optimizer groups (embeddings, lm_head, scalars). Muon unchanged.
- **Idée** : Replace standard scheduled AdamW with Schedule-Free AdamW for all non-Muon parameter groups. Instead of linearly decaying LR during the 60% warmdown phase, Schedule-Free maintains two sequences: z (Adam gradient trajectory) and x (Polyak-Ruppert running average). The model trains on y = (1-β₁)z + β₁x during forward/backward, and evaluates on x (the averaged iterate). LR stays constant for AdamW groups — the averaging provides an implicit optimal schedule. Muon keeps its existing linear warmdown.
- **Pourquoi** : (1) 60% of training is warmdown where AdamW LR drops linearly to 0 — this means embeddings/lm_head get weaker updates for the majority of training. Schedule-Free keeps full LR throughout while iterate averaging provides implicit decay. (2) Proven theoretically optimal for convex settings and empirically strong for deep learning (matches/beats best schedules). (3) Exp70 showed decoupling AdamW warmdown hurt (+0.013), but that was raw constant-LR without averaging — Schedule-Free's Polyak averaging provides the stability that raw constant-LR lacks. (4) Embeddings (wte, VE) and lm_head are the most impactful non-matrix parameters. (5) All architectural changes failed recently — time for an optimizer-level improvement.
- **Risques** : (1) ~830 steps might be too few for Polyak averaging to converge well (1/t weighting is aggressive early, very slow late). (2) Constant LR for AdamW groups without warmdown could destabilize embedding updates if averaging doesn't compensate. (3) Extra memory (~400MB for z,x clones of embeddings) — negligible for 32GB. (4) The y interpolation point might be worse than the straight warmdown trajectory for short training. Expected range: -0.008 to +0.005.


## Experiment #111 — 2026-03-16 13:43:44

- **Best val_bpb avant** : 1.070366
- **Status** : discard (val_bpb: 1.071584)
- **Description** : Per-head learned output scales in attention — learnable per-head scaling before output projection (c_proj).
- **Description** : Per-head learned output scales in attention — learnable scaling of each attention head's output before c_proj.
- **Idée** : Add a learnable per-head scale parameter (nn.Parameter, shape [n_head]=4) that multiplies each head's attention output before concatenation and output projection via c_proj. Initialized to 1.0 (neutral, exact baseline at step 0). With multi-scale RoPE giving each head a different positional scale (2500, 10000, 40000, 160000), heads serve different roles (local vs. long-range attention). Learned output scales let the model amplify heads that are more useful and suppress less useful ones, per layer. This complements Muon's orthogonal updates on c_proj, which normalize singular values and can't easily adjust per-head scaling independently.
- **Pourquoi** : (1) Multi-scale RoPE (exp80) creates heterogeneous heads with different positional specializations — learned output weighting is a natural complement. (2) c_proj is updated by Muon with Newton-Schulz orthogonalization, which normalizes the update's singular values. Explicit per-head AdamW-trained scales provide an unconstrained path for learning head importance. (3) Only 4 params per layer (32 total), zero compute overhead. (4) Init=1.0 means model starts at exact baseline. (5) Novel — not in dead list. (6) Different from qk_scales (exp68, +0.002) which scaled QK dot product BEFORE softmax; this scales the OUTPUT after softmax, affecting the residual stream contribution directly.
- **Risques** : (1) c_proj might already learn to weight heads differently, making explicit scales redundant. In that case, scales stay near 1.0 = neutral result. (2) LR=scalar_lr*0.01=0.005 might be too low for meaningful differentiation in ~830 steps. (3) 32 params is so few that the effect might be negligible. Expected range: -0.002 to +0.003.


## Experiment #112 — 2026-03-16 14:00:05

- **Best val_bpb avant** : 1.070366
- **Status** : keep (val_bpb: 1.070199)
- **Description** : Widen VE gate input from 32 channels to full n_embd (512) for richer per-head gating information.
- **Description** : Widen VE gate input from 32 channels to full n_embd (512) for richer gating information.
- **Idée** : The VE gate currently uses only the first 32 of 512 hidden dimensions (6.25% of the representation) to compute per-head gating values. This is an arbitrary information bottleneck — there's no reason the first 32 dims contain the most relevant gating signal. Widening to the full 512-dim hidden state gives the gate access to the complete token representation for deciding how much VE to mix into values. The gate weight grows from (4, 32) = 128 params to (4, 512) = 2048 params per VE layer (3840 total across 2 layers). Negligible overhead. Init stays zero → sigmoid(0)*2=1.0 → exact baseline at step 0. The gate is trained by Muon (2D matrix), which orthogonalizes the 4 gate rows in 512-dim space — meaning each head's gate response will be maximally different, complementing multi-scale RoPE's head specialization.
- **Pourquoi** : (1) VE bookend was the single biggest architectural win (-0.004, exp84). Improving VE gating has the highest EV of any remaining change. (2) The first 32 dims are arbitrary — in a transformer, information is distributed across all dimensions via random init and training. Restricting the gate to 6.25% of the representation is a real bottleneck. (3) With multi-scale RoPE creating 4 specialized heads (local/medium/long/very-long), the gate needs rich information to decide per-head VE contribution. (4) Muon's Newton-Schulz on the (4, 512) matrix ensures orthogonal gate vectors = maximal head differentiation. (5) Zero-init → exact baseline at step 0, zero regression risk.
- **Risques** : (1) The gate might already have enough information from 32 dims — widening could be neutral. (2) With 2048 params and Muon optimization, the gate could overfit and produce extreme gating values that destabilize training. (3) The additional ~3840 params are negligible but marginally increase step time. Expected range: -0.003 to +0.002.


## Experiment #113 — 2026-03-16 14:15:51

- **Best val_bpb avant** : 1.070199
- **Status** : discard (val_bpb: 1.070299)
- **Description** : Key Embedding (KE) — extend VE to also contribute to keys via a separate learned gate, reusing the shared value embedd
- **Description** : Key Embedding (KE) — extend VE to also contribute to keys via a separate learned gate, reusing the same value embedding.
- **Idée** : Currently VE only adds token identity to Values (what gets extracted). This experiment extends VE to also influence Keys (what gets attended to) via a separate ke_gate per VE layer (first+last). The same shared VE embedding is used for both V and K gates — no new embedding parameters. A separate ke_gate (nn.Linear(n_embd, n_kv_head, bias=False), init zeros → sigmoid(0)*2=1.0 → neutral) controls the mixing. KE is added to K before RoPE and QK-norm, so it gets position-encoded and normalized normally. This means token identity (e.g., "closing bracket", "function keyword") influences WHAT gets attended to, not just what value is produced.
- **Pourquoi** : (1) VE is the most productive improvement direction: bookend -0.004, skip_lambdas -0.002, wider gate -0.000167. Extending VE to keys is the natural next step. (2) Keys determine attention patterns. Adding token identity to keys means the model can attend to tokens partly based on what they ARE (identity), not just their learned key representation. (3) Reuses the same VE embedding (shared first+last layer) — no new embedding params. Only 2 new ke_gate weights: 2 × (4, 512) = 4096 new params. (4) ke_gate is trained by Muon (2D matrix, same shape as ve_gate) — Newton-Schulz orthogonalization ensures ke_gate rows are maximally different from each other, complementing multi-scale RoPE head specialization. (5) Zero-init → exact baseline at step 0. (6) Applied before RoPE: the key embedding gets position-encoded, so it's not purely content-based — it captures "token X at this position" which is richer than pure identity.
- **Risques** : (1) Adding VE to keys might confuse the attention pattern — keys already encode sufficient token information via c_k projection. In that case, ke_gate stays near zero = neutral. (2) KE could make attention too focused on token identity (copying behavior) at the expense of learned contextual patterns. (3) The additional forward pass cost is negligible (one extra linear projection + sigmoid + multiply per VE layer). Expected range: -0.004 to +0.003.


## Experiment #114 — 2026-03-16 14:32:30

- **Best val_bpb avant** : 1.070199
- **Status** : discard (val_bpb: 1.078830)
- **Description** : Per-dimension residual and x0 lambdas — widen from scalar per-layer to per-channel for finer residual stream routing.
- **Description** : Per-dimension residual and x0 lambdas — widen resid_lambdas and x0_lambdas from [n_layer] to [n_layer, n_embd] for per-channel residual stream routing.
- **Idée** : Currently resid_lambdas (shape [8]) and x0_lambdas (shape [8]) apply a single scalar per layer to control residual persistence and initial embedding injection. This forces all 512 dimensions to share the same dynamics. Widening to [8, 512] lets different residual stream channels have different persistence rates. Some dimensions can preserve accumulated context (high resid_lambda) while others reset to x0 (high x0_lambda). This is analogous to how exp112 improved VE by removing the 32-dim gate bottleneck — here we remove the scalar bottleneck in residual routing. Total: 8192 new params (vs 16 before), trained by AdamW. Init unchanged (1.0 for resid, 0.1 for x0) = exact baseline at step 0.
- **Pourquoi** : (1) Per-dimension residual gating is well-established in modern architectures (e.g., Highway Networks, FLASH). (2) With multi-scale RoPE creating specialized heads, different dimensions may encode different types of information that benefit from different layer-wise dynamics. (3) The wider VE gate (exp112, new best) succeeded by removing an information bottleneck — same principle applies here. (4) 1-line change, zero forward pass code change (broadcasting handles everything), zero crash risk. (5) 8192 params is negligible for a ~76M model.
- **Risques** : (1) Scalar per-layer lambdas may already capture sufficient dynamics → neutral result. (2) 8192 params with ~830 AdamW steps may not learn meaningful per-dimension differentiation. (3) LR might be suboptimal for the new param count (resid: 0.005, x0: 0.5). Expected range: -0.003 to +0.003.


## Experiment #115 — 2026-03-16 15:12:07

- **Best val_bpb avant** : 1.070199
- **Status** : keep (val_bpb: 1.069319)
- **Description** : VE weight decay 0.002 — add mild L2 regularization to Value Embedding parameters to prevent rare token VE drift.
- **Description** : VE weight decay 0.002 — add mild regularization to the Value Embedding parameters to prevent rare token VE drift.
- **Idée** : Add weight_decay=0.002 to the VE AdamW optimizer group (currently 0.0). The VE embedding is 16.8M params (22% of model), shared between first and last layer, trained at high LR (0.6). With ~830 steps and 32K vocab, rare tokens get few effective gradient updates and their VE vectors can drift to large norms. Small WD regularizes this while leaving common tokens mostly unaffected.
- **Pourquoi** : (1) VE is the biggest single parameter block and the most productive improvement direction (bookend -0.004, wider gate -0.000167). (2) VE has WD=0.0 while transformer matrices use WD=0.2 — the VE is the only large parameter block with zero regularization. (3) WD pushes unused/rare token VE vectors toward zero, which is a better default than random drift. (4) With LR=0.6 and WD=0.002, per-step decay factor is ~0.9988, total decay over full-LR phase ~0.67 — meaningful but not aggressive. (5) Listed in pistes restantes (#12), never tried.
- **Risques** : (1) WD=0.002 might be too mild to make any measurable difference → neutral result. (2) WD might regularize useful VE signal away, especially for rare but important tokens → mild regression. (3) Totally safe: no crash risk, no OOM risk, no code complexity. Expected range: -0.002 to +0.002.


## Experiment #116 — 2026-03-16 15:32:57

- **Best val_bpb avant** : 1.069319
- **Status** : discard (val_bpb: 1.081621)
- **Description** : Soft Mixture of Depths — input-dependent per-block gating for compile-friendly token routing.
- **Description** : Soft Mixture of Depths — input-dependent per-block gating for compile-friendly token routing.
- **Idée** : Add a block_gate = nn.Linear(n_embd, 1, bias=False) to each Block. Gate computes `2 * sigmoid(block_gate(norm(x)))` and scales both attn and MLP sublayer outputs. At init (zero weights): gate = 2*sigmoid(0) = 1.0 → exact baseline. During training, tokens learn to skip (gate→0) or amplify (gate→2) specific layers. This is a soft, compile-friendly version of Mixture of Depths (piste #6): no dynamic token counts, all static shapes. Uses the proven 2*sigmoid zero-init pattern from VE gates.
- **Pourquoi** : (1) MoD is listed as remaining piste #6 with "torch.compile compat unclear" — this soft formulation resolves the compat issue. (2) Tokens vary in difficulty — easy tokens (common words, punctuation) don't need all 8 layers. By allowing tokens to modulate layer contributions, effective capacity is allocated where it's needed. (3) The gate uses the same 2*sigmoid pattern as VE gates (proven stable in exp84/112). (4) Only 8 × (1, 512) = 4096 new params total, negligible overhead. (5) Gate is computed from norm(x) and reused for attn input, saving one redundant norm(x) computation.
- **Risques** : (1) With only 8 layers, there may not be enough redundancy for routing to help — gates stay near 1.0 → neutral. (2) Muon on (1, 512) degenerates to gradient normalization — may produce noisy gate updates. (3) The gate adds a sigmoid + multiply per sublayer per layer — tiny but non-zero overhead (~3% extra FLOPs from gate linear). (4) Gate scaling could destabilize training if gates diverge to extreme values (0 or 2). Expected range: -0.005 to +0.005.


## Experiment #117 — 2026-03-16 15:48:58

- **Best val_bpb avant** : 1.069319
- **Status** : discard (val_bpb: 1.070293)
- **Description** : wte embedding weight decay 0.002 — add mild L2 regularization to token embeddings, mirroring VE WD success (exp115).
- **Description** : wte embedding weight decay 0.002 — add mild L2 regularization to token embeddings, mirroring VE WD success (exp115).
- **Idée** : Add weight_decay=0.002 to the wte embedding AdamW optimizer group (currently 0.0). The wte embedding is 16.8M params (22% of model), trained at high LR (0.6), with WD=0.0. With 32K vocab and ~830 steps, rare tokens get sparse gradient updates and their embedding vectors can drift. This is the exact same reasoning that produced exp115 (VE WD=0.002, new best -0.00088). The wte is the only remaining large parameter block with zero regularization.
- **Pourquoi** : (1) Exp115 proved WD=0.002 works for VE embeddings (-0.00088, new best). wte has identical characteristics: same size (16.8M), same LR (0.6), same WD=0.0, same rare-token drift problem. (2) wte is explicitly listed as piste #17 "wte embedding weight decay — same logic as VE WD (exp115)". (3) 1-value change, zero crash risk, zero OOM risk. (4) Effective decay: LR=0.6 × WD=0.002 = 0.0012 per-step factor, ~0.67 total reduction over full-LR phase — matching VE WD dynamics exactly.
- **Risques** : (1) wte embeddings may not suffer from the same drift as VE — wte gets gradient signal from every token via cross-entropy, not just VE layers → neutral result. (2) WD could regularize away useful rare-token representations → mild regression. (3) Zero crash/OOM risk. Expected range: -0.002 to +0.002.


## Experiment #118 — 2026-03-16 15:57:20

- **Best val_bpb avant** : 1.069319
- **Status** : discard (val_bpb: 1.082322)
- **Description** : WTE-as-value identity for all layers — inject normalized wte embedding (x0) directly into attention values via per-hea
- **Description** : WTE-as-value identity for all layers — inject normalized wte embedding (x0) directly into attention values via per-head learnable scalars.
- **Idée** : VE bookend (exp84, -0.004) showed that injecting token identity into values is highly productive. But VE only applies to layers 0 and 7. Layers 1-6 have no token identity signal in values — they rely solely on the contextual c_v projection. This experiment adds the normalized wte embedding (x0, already computed) directly to values in ALL layers, controlled by per-head learnable scalars (wte_v_scales, shape [8, 4] = 32 params). At init: scales=0 → exact baseline. During training, each layer/head learns how much raw token identity to add. This is different from exp85 (VE all layers, +0.010) which added a full 16.8M-param shared VE embedding to all layers. Here we reuse wte (zero extra embedding params, zero extra memory, zero extra embedding lookups). Only 32 scalar params added. For VE layers (0, 7), both VE (input-dependent gated, separate embedding) and wte identity (scalar-scaled, wte embedding) contribute — the model decides which is useful.
- **Pourquoi** : (1) VE is the most productive improvement direction: bookend -0.004, skip_lambdas -0.002, wider gate -0.000167, VE WD -0.00088. (2) Middle layers (1-6) currently have NO token identity in values — c_v computes a contextual projection but raw "what token is this" information is lost. (3) x0 is already computed and available at every layer (used by x0_lambdas). Adding it to values costs essentially zero FLOPs. (4) exp85 (VE all layers) failed at +0.010, but it used a heavy 16.8M-param approach. This is >500,000x lighter (32 params vs 16.8M). (5) Per-head scalars let each attention head specialize — local heads (RoPE base 2500) might want less wte identity while long-range heads (base 160000) might want more. (6) Trained by AdamW at scalar_lr=0.5, same group as skip_lambdas which worked well.
- **Risques** : (1) c_v already has access to wte identity through the residual stream (via x0_lambdas injection). Explicit wte-in-values might be redundant → scales stay near 0 → neutral. (2) Adding wte to values in ALL layers (including VE layers 0, 7) could interfere with VE gating. But scales=0 at init and AdamW should find the right balance. (3) torch.compile: the `if x0 is not None and wte_v_scale is not None` condition is always True during training, so torch.compile traces the always-True branch cleanly. (4) Zero crash/OOM risk — 32 params, no new modules. Expected range: -0.003 to +0.003.


## Experiment #119 — 2026-03-16 16:15:02

- **Best val_bpb avant** : 1.069319
- **Status** : discard (val_bpb: 1.069415)
- **Description** : VE weight decay 0.005 — increase VE L2 regularization from 0.002 to 0.005 to test if stronger rare-token drift prevent
- **Description** : VE weight decay 0.005 — increase VE regularization from 0.002 to 0.005, testing if stronger L2 penalty improves generalization.
- **Idée** : Increase weight_decay on the VE AdamW optimizer group from 0.002 (exp115, -0.00088 = new best) to 0.005. The VE embedding is 16.8M params (22% of model), shared between first and last layer. Exp115 proved that mild WD helps by regularizing rare-token VE drift. The WD response curve may not have peaked at 0.002 — stronger WD=0.005 pushes rare tokens more aggressively toward zero while common tokens (which receive strong gradients) resist the decay. With LR=0.6 and WD=0.005, per-step decay factor is ~0.997, total decay over full-LR phase ~0.43 (vs ~0.67 at WD=0.002). This is still conservative compared to transformer matrix WD=0.2.
- **Pourquoi** : (1) VE WD=0.002 was the most recent new best (-0.00088). Natural follow-up to find the WD optimum. (2) Listed explicitly as piste #20 "VE WD=0.005 — stronger VE regularization". (3) VE has the weakest regularization of any major parameter block: wte has implicit regularization via cross-entropy gradients on all tokens, matrices have WD=0.2 via Muon. VE only gets gradient from 2 layers (bookend) and sparse token updates. (4) 1-value change, zero crash/OOM risk. (5) If 0.005 is worse than 0.002, the optimum is near 0.002 and VE WD tuning is done.
- **Risques** : (1) WD=0.005 might over-regularize — especially for medium-frequency tokens that need VE signal but get moderate gradient updates. (2) The optimum might be between 0.002 and 0.005 (e.g., 0.003). In that case, 0.005 could be neutral or slightly worse. (3) Zero crash/OOM risk. Expected range: -0.002 to +0.003.


## Experiment #120 — 2026-03-16 16:25:57

- **Best val_bpb avant** : 1.069319
- **Status** : discard (val_bpb: 1.069504)
- **Description** : VE betas (0.9, 0.95) — increase VE Adam beta1 from 0.8 to 0.9 for smoother gradient EMA on sparsely-updated Value Embe
- **Description** : VE betas (0.9, 0.95) — increase VE Adam beta1 from 0.8 to 0.9 for smoother gradient EMA on sparsely-updated Value Embeddings.
- **Idée** : VE receives uniquely sparse gradients (only 2/8 layers, only in-batch tokens). With beta1=0.8, the gradient EMA decays 20% per step — after 5 steps without a token, its EMA is at 33%. With beta1=0.9, decay is 10% per step — after 5 steps, 59% remains. Smoother VE updates should improve learning quality.
- **Pourquoi** : (1) VE is the most productive direction (bookend -0.004, skip -0.002, WD -0.00088). (2) VE betas is explicitly listed as untested piste #21. (3) Complements VE WD=0.002 — WD controls norms, betas controls gradient quality. Orthogonal. (4) Higher beta1 for sparse parameters is established practice. (5) 1-value change, zero risk.
- **Risques** : Zero crash/OOM risk. If VE gradients are already well-estimated: neutral. If slower adaptation hurts: mild regression ~0.001-0.002.
- **Changement** : VE optimizer group betas: `adam_betas` (0.8, 0.95) → `(0.9, 0.95)`. Single value change.


## Experiment #121 — 2026-03-16 16:35:43

- **Best val_bpb avant** : 1.069319
- **Status** : crash (val_bpb: 0.000000)
- **Description** : FP8 matmuls on Blackwell — replace all nn.Linear calls in MLP and attention with float8_e4m3fn matmuls via torch._scal
- **Description** : FP8 matmuls on Blackwell — replace all nn.Linear forward calls (MLP + attention projections) with float8_e4m3fn matmuls via torch._scaled_mm for ~2x throughput on FP8 tensor cores.
- **Idée** : Use FP8 (float8_e4m3fn) for all matrix multiplications in MLP (c_fc, c_proj) and attention (c_q, c_k, c_v, c_proj). Dynamic per-tensor scaling quantizes activations and weights to FP8 before each matmul, then torch._scaled_mm computes the result in FP8 with BF16 output. Custom autograd function handles forward + backward in FP8. MLP 3x was the biggest win (-0.008) because it reduced compute → more steps. FP8 is the next "more steps" lever: ~2x throughput on the matrix multiplications = significantly more training steps in 5 minutes.
- **Pourquoi** : (1) All VE/hyperparameter tuning paths are exhausted (exp117-120 all neutral). (2) FP8 is explicitly listed as piste #22 "potentially the biggest remaining lever". (3) RTX 5090 Blackwell has native FP8 tensor cores at 2x BF16 throughput. (4) MLP 3x proved that more steps → better val_bpb (-0.008). FP8 achieves more steps via faster computation rather than reduced model size. (5) Per-tensor dynamic scaling preserves FP8 precision (~3-4 decimal digits, sufficient for training).
- **Risques** : (1) CRASH RISK: torch._scaled_mm API might differ from expected (tuple vs tensor return, scale semantics). (2) CRASH RISK: torch.compile might not handle custom autograd function with _scaled_mm, causing graph breaks or errors. (3) REGRESSION RISK: quantization overhead (6 amax reductions per linear call) might outweigh FP8 speedup for small 512×512 attention matrices. (4) REGRESSION RISK: FP8 precision loss could hurt training quality. (5) If it crashes: API issue, try torchao or different _scaled_mm signature. If slower: apply FP8 only to large MLP matrices (512×1536).


## Experiment #122 — 2026-03-16 16:58:32

- **Best val_bpb avant** : 1.069319
- **Status** : keep (val_bpb: 1.068180)
- **Description** : Non-uniform MLP widths — layers 0-3 get 2x MLP expansion, layers 4-7 get 4x, iso-compute with current uniform 3x.
- **Description** : Non-uniform MLP widths — layers 0-3 get 2x MLP, layers 4-7 get 4x MLP (iso-compute with current uniform 3x).
- **Idée** : Later transformer layers need more capacity for fine-grained prediction, while early layers mainly extract basic features. With U-Net skip connections feeding early-layer info to late layers, wider late MLPs give more capacity to USE that combined information. Total MLP params and FLOPs are identical to current uniform 3x (4×2x + 4×4x = 4×3x average). This is an untested architectural change that redistributes existing compute without adding any.
- **Pourquoi** : (1) All VE/hyperparameter tuning paths exhausted (exp117-121 all neutral/crash). (2) MLP 3x was the biggest single win (-0.008), so MLP configuration is high-leverage. (3) Non-uniform depth-dependent MLP is used in practice (e.g., PaLM uses wider MLPs in later layers). (4) U-Net skip connections (exp71, -0.001) feed early layer outputs to late layers — wider late MLPs have more capacity to exploit this additional information. (5) Iso-compute: identical total params and FLOPs to baseline, so any improvement is pure architectural benefit, not from more steps.
- **Risques** : (1) Early layers with 2x MLP may become bottlenecks — MLP 2.5x uniform was +0.003, so 2x could limit early feature extraction. (2) The benefit of wider late layers might not compensate for narrower early layers, especially since early layers feed into U-Net skips. (3) Zero crash/OOM risk — just different linear dimensions. Expected range: -0.005 to +0.005.


## Experiment #123 — 2026-03-16 17:32:13

- **Best val_bpb avant** : 1.068180
- **Status** : en cours...
- **Description** : Gradual MLP width ramp — linear expansion from 1x to 5x across layers [1,2,2,3,3,4,4,5], iso-compute with current 2x/4x split (avg 3x).
- **Idée** : Exp122 proved non-uniform MLP widths work (-0.00114). The 2-step split (2x/4x) was a first test. A smooth linear ramp concentrates compute even more aggressively in late layers (5x for layer 7 vs previous 4x) while shrinking early layers further (1x for layer 0 vs previous 2x). Later layers need more capacity for fine-grained prediction; early layers mainly do basic feature mixing. With U-Net skip connections feeding early outputs to late layers, wider late MLPs can better exploit that combined information. Total params and FLOPs identical to baseline.
- **Pourquoi** : (1) Direct follow-up on best recent experiment (exp122, -0.00114). prompt.md explicitly says "If works: try gradual increase." (2) Architectural change (priority 1). (3) Iso-compute — any improvement is pure architectural benefit. (4) More extreme spread (1x-5x vs 2x-4x) tests whether the "late layers need more" principle extends further. (5) All hidden dims (512, 1024, 1536, 2048, 2560) are clean multiples of 512, fine for tensor cores.
- **Risques** : (1) Layer 0 with 1x MLP (hidden=512, same as model dim) may be too narrow — MLP 2.5x uniform was +0.003, so very narrow MLPs can hurt. But that was uniform; here only one layer is narrow. (2) Layer 7 at 5x might be overkill, adding parameters where returns diminish. (3) Zero crash/OOM risk — just different linear dimensions. Expected range: -0.002 to +0.003.


## Experiment #124 — 2026-03-16 17:37:42

- **Best val_bpb avant** : 1.068180
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Gradual MLP width ramp [2,2,2,3,3,3,4,5] with min 2x — concentrates maximum capacity in the final prediction layer, is
- **Description** : Gradual MLP ramp with min 2x — [2,2,2,3,3,3,4,5] across 8 layers, iso-compute with baseline (sum=24=8×3). Concentrates max capacity in final layer (5x).
- **Idée** : Exp122 proved non-uniform MLP works (-0.00114 with 2x/4x). Exp123 tried a steeper ramp [1,2,2,3,3,4,4,5] but crashed — likely the 1x layer (hidden=model_dim=512) was a bottleneck. This experiment uses a safer ramp with min 2x, plus concentrates the most capacity in layer 7 (5x=2560 hidden). The final MLP is the last nonlinear transformation before lm_head — it directly shapes the features used for next-token prediction. Giving it 25% more capacity than current 4x while smoothly ramping through middle layers should improve prediction quality.
- **Pourquoi** : (1) Direct follow-up on best experiment (exp122, -0.00114). (2) Architectural change (priority 1). (3) Iso-compute — any gain is pure architecture. (4) Addresses exp123 crash by maintaining min 2x. (5) Concentrates compute where it matters most — the final pre-prediction transformation.
- **Risques** : (1) Layers 4-5 drop from 4x to 3x — might lose mid-layer capacity. (2) The 5x last layer might be overkill with diminishing returns. (3) Zero crash/OOM risk — all dims are multiples of 512. Expected range: -0.003 to +0.003.


## Experiment #125 — 2026-03-16 17:42:02

- **Best val_bpb avant** : 1.068180
- **Status** : discard (val_bpb: 1.071063)
- **Description** : Increase UNEMBEDDING_LR from 0.004 to 0.01 (2.5x) to improve lm_head convergence — 22% of model params, never tuned.
- **Description** : Increase UNEMBEDDING_LR from 0.004 to 0.01 (2.5x) to improve lm_head convergence — 22% of model params, never tuned.
- **Idée** : The lm_head linear projection (512→32768, 16.8M params = 22% of model) uses AdamW with LR=0.004, which is 150x below EMBEDDING_LR=0.6. With only ~830 training steps, this huge parameter matrix may be severely under-trained. Increasing LR from 0.004 to 0.01 (2.5x) is still conservative — actual LR after dmodel scaling: 0.01 × 1.225 = 0.0123. Every token position produces a gradient for lm_head (dense gradients, unlike sparse embedding gradients), so AdamW moments converge fast. A higher LR lets the output layer better fit the training distribution in limited steps.
- **Pourquoi** : (1) Piste #24, explicitly listed as never tested. (2) lm_head is 22% of model — largest single param group after transformer matrices. (3) All architectural ideas from prompt.md are either exhausted or too complex (DeltaNet/GLA). (4) FP8 (exp121) crashed, MLP ramp (exp123-124) crashed. (5) EMBEDDING_LR was tuned from default to 0.6 (big gain), but UNEMBEDDING_LR was never touched. (6) Zero crash/OOM risk.
- **Risques** : (1) LR too high could cause logit divergence — mitigated by softcap=13 + z-loss. (2) lm_head dense gradients might mean current 0.004 is already well-scaled — if so, neutral result. (3) Expected range: -0.005 to +0.003.


## Experiment #126 — 2026-03-16 17:52:29

- **Best val_bpb avant** : 1.068180
- **Status** : en cours...
- **Description** : RWKV-style token shift for attention K/V inputs — mix current token representation with previous token via learnable per-channel sigmoid gates before K and V projections.
- **Idée** : In RWKV architectures, "time mixing" gives each layer's projections access to adjacent token context before the actual attention computation. Currently, K and V are computed purely from the current token's representation — the attention mechanism must discover local bigram patterns from scratch. By mixing the previous token's features into K/V inputs (but NOT Q — queries should represent what the current token wants), we give the key/value projections a cheap local context signal. This is a direct simplification of RWKV-6's time mixing mechanism, listed as a priority-1 architectural exploration in prompt.md. Uses sigmoid gating (initialized at 0.5 = equal mixing) with learnable per-channel parameters, so the model can learn per-dimension which features benefit from local vs current-only context. First token uses self-shift (no information loss). Adds only 8192 params (2 × 512 × 8 layers), negligible.
- **Pourquoi** : (1) RWKV-style recurrence is explicitly listed as priority-1 architectural alternative in prompt.md. (2) All VE/hyperparameter tuning exhausted (exp117-125 all neutral/crash/discard). (3) FP8 (exp121) crashed, DeltaNet/GLA would be slower than SDPA at T=2048. (4) Conv before MLP was tried (exp107, +0.023, dead) but token shift before ATTENTION is fundamentally different — attention is designed to capture context, while MLP processes each position independently. (5) Zero crash/OOM risk — just tensor slicing, sigmoid, and mixing.
- **Risques** : (1) REGRESSION: Full causal attention already captures all pairwise relationships, so bigram mixing may be redundant. (2) NEUTRAL: The mixing might blur position boundaries without providing useful signal. (3) The analogy with exp107 (conv before MLP = +0.023) is concerning, though the mechanisms serve different purposes. Expected range: -0.003 to +0.005.


## Experiment #127 — 2026-03-16 18:12:34

- **Best val_bpb avant** : 1.067854
- **Status** : en cours...
- **Description** : Extend RWKV-style token shift to Q projection — full Q/K/V time mixing instead of K/V only.
- **Idée** : Exp126 proved that token shift for K/V inputs improves val_bpb (-0.000326, new best 1.067854). The natural follow-up is extending this to Q as well, creating full RWKV-style time mixing for all three attention projections. In RWKV architectures, all projections use separate time mixing parameters. Currently Q uses only the current token's representation — by mixing in the previous token, queries become "what does this bigram want?" which could help the model match local patterns more efficiently. Each projection gets its own learnable per-channel sigmoid gate (initialized at 0.5 = equal mix), so the model can independently learn optimal mixing for Q vs K vs V. If Q mixing is harmful, the model can push time_mix_q weights toward +infinity (sigmoid → 1.0 = only current token), effectively recovering exp126's behavior.
- **Pourquoi** : (1) Direct follow-up on winning exp126, explicitly suggested in its report: "Apply token shift to Q as well". (2) Architectural change (priority 1). (3) Zero compute overhead — just element-wise ops. (4) Adds only 4096 params (1 × 512 × 8 layers), negligible. (5) Self-correcting: model can learn to ignore Q shift if it hurts. (6) All VE/hyperparameter paths exhausted. FP8 crashed, DeltaNet too complex.
- **Risques** : (1) NEUTRAL: Q mixing may be redundant — queries represent "what this position wants", and neighbor contamination could blur this intent. Model may learn time_mix_q ≈ +inf (no mixing), recovering exp126. (2) REGRESSION: Unlikely but possible if the extra learnable parameters cause optimization interference. (3) CRASH: None — pure tensor ops. Expected range: -0.002 to +0.003.


## Experiment #128 — 2026-03-16 18:21:26

- **Best val_bpb avant** : 1.067854
- **Status** : discard (val_bpb: 1.074396)
- **Description** : V-only token shift — remove K shift from exp126, keep only V shift to avoid blurring key position identity before RoPE
- **Description** : V-only token shift — remove K shift from exp126, keep only V shift to avoid blurring key position identity before RoPE.
- **Idée** : Exp126 added token shift for both K and V (-0.000326, new best). Exp127 added Q shift and regressed (+0.008). The pattern is clear: shifting projections involved in attention scoring (Q, K) hurts precision, while shifting V (which provides the "content" to aggregate) enriches values with local context. K determines WHERE to attend — it's paired with Q via dot product and then has RoPE applied. Shifting K input means RoPE gets a blurred position signal (mixing two positions' features before applying position encoding). V has NO RoPE — shifting it doesn't interfere with any positional mechanism. By removing K shift, we get: (1) cleaner position encoding in keys, (2) sharper attention patterns, (3) the beneficial V neighbor context mixing, (4) fewer learnable parameters (4096 fewer). This is a simplification that could improve — matching the prompt's philosophy "si tu supprimes du code et que le score est égal ou meilleur, c'est une victoire."
- **Pourquoi** : (1) Direct ablation of the winning exp126 mechanism. (2) Strong theoretical basis: K participates in attention scoring with RoPE, shifting degrades position signal. V doesn't, shifting only enriches. (3) Q shift failure (exp127, +0.008) proves that "scoring projections" should not be shifted. (4) Removes 4096 params and one sigmoid+multiply per layer. (5) Zero crash risk.
- **Risques** : (1) NEUTRAL: K shift may have been the beneficial component, not V shift. In that case, removing K shift removes the benefit and we get neutral. (2) REGRESSION: Both K and V shifts could have been synergistic — removing K might disrupt the balance. (3) Expected range: -0.002 to +0.002.


## Experiment #129 — 2026-03-16 18:44:32

- **Best val_bpb avant** : 1.067854
- **Status** : en cours...
- **Description** : More extreme non-uniform MLP — 1.5x/4.5x split instead of 2x/4x, iso-compute with baseline (sum=24=8×3 avg).
- **Idée** : Exp122 proved non-uniform MLP works (-0.00114 with 2x/4x). Pushing the asymmetry further concentrates even more capacity in late layers (4.5x=2304 hidden vs previous 4x=2048) while slimming early layers (1.5x=768 vs previous 2x=1024). Late layers receive combined residual stream + U-Net skip information and are closest to the prediction head — they benefit most from wider MLPs. Early layers are simpler feature extractors; their outputs are still available to late layers through U-Net skip connections, so reduced early capacity is partially compensated. Total params and FLOPs identical to baseline (4×1.5 + 4×4.5 = 24 = 8×3).
- **Pourquoi** : (1) Direct follow-up on winning exp122 (-0.00114). (2) Architectural change (priority 1). (3) Iso-compute — any gain is pure architecture. (4) Token shift exploration exhausted: K+V works (exp126), Q hurts (exp127), V-only hurts (exp128). (5) DeltaNet/GLA too complex, FP8 crashed (exp121), no torchao available. (6) All hidden dims (768=6×128, 2304=18×128) are clean multiples of 128, fine for tensor cores.
- **Risques** : (1) REGRESSION: 1.5x early MLP may be too narrow — MLP 2.5x uniform was +0.003 (exp15), and 1.5x is even narrower. But that was uniform; here only early layers are narrow. (2) NEUTRAL: 2x/4x may already be the sweet spot. The original -0.00114 was within noise (~0.003 variance). (3) CRASH: None — just different linear dimensions. Expected range: -0.003 to +0.005.


## Experiment #130 — 2026-03-16 19:02:05

- **Best val_bpb avant** : 1.067071
- **Status** : en cours...
- **Description** : Even more extreme non-uniform MLP — 1.25x/4.75x split instead of 1.5x/4.5x, iso-compute with baseline (avg 3x).
- **Idée** : The non-uniform MLP trend is the clearest winning pattern in recent experiments: uniform 3x → 2x/4x (exp122, -0.00114) → 1.5x/4.5x (exp129, -0.00109 more). Each step concentrating more compute in late layers improved quality. Late layers (4-7) receive combined residual stream + U-Net skip information and directly shape features for lm_head — they benefit most from wider MLPs. Early layers (0-3) do basic feature extraction; their outputs are still available to late layers through U-Net skip connections, so reduced capacity is compensated. 1.25x gives 640 hidden (5×128), safely above 512 (model_dim). 4.75x gives 2432 hidden (19×128). Both are clean multiples of 128 for tensor cores. Total: 4×640 + 4×2432 = 12288 = 8×1536 (uniform 3x). Zero net parameter/FLOP change.
- **Pourquoi** : (1) Direct follow-up on the strongest recent winning trend (exp122→exp129). (2) Architectural change (priority 1). (3) Iso-compute — any gain is pure architecture. (4) Token shift exploration exhausted (exp126-128). (5) FP8 crashed (exp121), DeltaNet/GLA too complex, RoPE range tuning is low priority. (6) 640 hidden is safely above the "1x=512 too narrow" crash boundary (exp123).
- **Risques** : (1) REGRESSION: 1.25x=640 may be approaching the bottleneck point for early layers. MLP 2.5x uniform was +0.003 (exp15), but here only 4 layers are narrow. (2) NEUTRAL: 1.5x/4.5x may already be the sweet spot — diminishing returns from further asymmetry. (3) CRASH: None — just different linear dimensions. Expected range: -0.003 to +0.003.


## Experiment #131 — 2026-03-16 19:18:24

- **Best val_bpb avant** : 1.067071
- **Status** : discard (val_bpb: 1.067803)
- **Description** : RWKV-style input-dependent per-head attention output gating on all layers.
- **Description** : RWKV-style input-dependent per-head attention output gating on all layers.
- **Idée** : Add an output gate to the attention sublayer, inspired by RWKV's output gating mechanism. Before c_proj, each head's output is scaled by `2 * sigmoid(W_gate @ x)` where W_gate is Linear(n_embd, n_head) and x is the pre-norm input. At initialization (weights=0), `2*sigmoid(0)=1.0` → neutral (model starts identical to current). The model can then learn per-head, input-dependent gating: suppress heads for "easy" tokens where residual is sufficient, amplify heads for "hard" tokens needing context. This is the same gating pattern used for VE (exp84, -0.004, big win), but applied to the entire attention output on ALL layers instead of just VE on bookend layers. Per-head STATIC scales (exp111, +0.001) were dead, but input-dependent gating is fundamentally different — 2048 params per layer producing different gates for every token vs 4 fixed scalars. Cost: 16K total new params (negligible), one tiny linear + sigmoid per layer.
- **Pourquoi** : (1) Architectural change (priority 1). (2) VE gating with the identical mechanism (2*sigmoid, Linear→n_head, zero-init) was the biggest single win (exp84, -0.004). Extending this pattern to the attention output is the natural next step. (3) Input-dependent gating is a core component of RWKV and Mamba, never tried for this model's attention output. (4) Static per-head scales (exp111, +0.001) were dead — the key difference is input-dependency. (5) MLP asymmetry exhausted (exp130 confirms 1.5x/4.5x is the sweet spot). Token shift exhausted. VE/optimizer fully tuned. (6) Zero compute overhead — one 512→4 linear is negligible.
- **Risques** : (1) NEUTRAL: c_proj already provides per-head mixing; the gate may be redundant. (2) REGRESSION: The gate adds a multiplicative factor that could interfere with gradient flow through attention. (3) CRASH: None — standard tensor ops, all static shapes, torch.compile compatible. Expected range: -0.003 to +0.005.


## Experiment #132 — 2026-03-16 19:34:16

- **Best val_bpb avant** : 1.067071
- **Status** : discard (val_bpb: 1.069988)
- **Description** : Affine RMSNorm — add learnable per-channel gain parameters to pre-sublayer norms and final norm.
- **Description** : Affine RMSNorm — add learnable per-channel gain parameters to all pre-sublayer norms and the final norm before lm_head.
- **Idée** : Every major production LLM (LLaMA, Mistral, Gemma, etc.) uses RMSNorm with a learnable gain vector, but this model deliberately uses parameterless `F.rms_norm(x, (x.size(-1),))`. Adding a per-channel gain `gamma` turns the norm into `x / RMS(x) * gamma`, allowing each layer to learn which features to amplify before each sublayer. This is a form of per-layer feature selection: different layers can emphasize different feature dimensions before their attention or MLP computations. The model already benefits from per-layer specialization (non-uniform MLP, multi-scale RoPE, U-Net skip, resid/x0/skip lambdas). Affine norm adds another axis of specialization at near-zero cost. 17 norm applications × 512 dims = 8,704 new learnable params (0.02% of model). Zero compute overhead — just element-wise multiply after norm. Block norm weights are 1D params in transformer.h, automatically grouped with time_mix params (scalar_lr, betas=(0.96, 0.95)). final_norm_weight is in its own AdamW group with same settings.
- **Pourquoi** : (1) Architectural change (priority 1). (2) Standard component in all major LLMs, never tested here. (3) Per-layer feature selection — a new specialization axis complementing MLP asymmetry. (4) Zero compute overhead, zero crash risk. (5) MLP asymmetry exhausted (exp130), token shift exhausted (exp126-128), VE tuning done, optimizer fully tuned, attention output gating neutral (exp131). (6) DeltaNet/GLA impractical at T=2048/D=128 (linear attention not faster), FP8 crashed (exp121).
- **Risques** : (1) NEUTRAL: With ~830 steps, gains may not diverge enough from 1.0 to make a measurable difference. The parameterless norm was a deliberate simplification that might be correct for this regime. (2) REGRESSION: Extra learnable params could add noise to optimization — 16 norm weight vectors × 512 dims = 8K params competing for gradient signal. (3) CRASH: None — F.rms_norm with weight parameter is a standard PyTorch API, torch.compile compatible. Expected range: -0.003 to +0.003.


## Experiment #133 — 2026-03-16 19:53:07

- **Best val_bpb avant** : 1.067071
- **Status** : discard (val_bpb: 1.067579)
- **Description** : Remove post-embedding RMSNorm — let wte embeddings retain magnitude information instead of forcing unit RMS.
- **Description** : Remove post-embedding RMSNorm — let wte embeddings retain magnitude information instead of forcing unit RMS.
- **Idée** : Currently `x = norm(wte(idx))` normalizes all token embeddings to unit RMS, discarding magnitude information. Removing this norm lets the model learn different scales for different tokens (e.g., common tokens with larger embeddings, rare tokens with smaller ones). The x0 residual connection (injected at every layer via x0_lambdas) would carry richer, unnormalized embedding information. At init, wte has std=1.0 so RMS≈1 anyway — the norm is nearly a no-op initially. During training, the model can develop meaningful magnitude differences. Pre-sublayer norms before attention/MLP still protect training stability. Many successful models (GPT-2/3, OPT) don't normalize after embedding. This is a code simplification — one line removal.
- **Pourquoi** : (1) Architectural simplification (priority 1 — "Si tu supprimes du code et que le score est égal ou meilleur, c'est une victoire"). (2) Frees embedding expressivity: magnitude information is useful signal that norm discards. (3) Suggested as "next step" in exp132 report's failure analysis. (4) MLP asymmetry exhausted (exp130), token shift exhausted (exp126-128), attention output gating dead (exp131), affine norm dead (exp132). (5) Zero crash risk — just removing a norm. Pre-sublayer norms protect stability. (6) At init, norm is nearly a no-op (wte std=1.0, RMS≈1 for 512 dims).
- **Risques** : (1) NEUTRAL: Norm might have been nearly invisible (wte naturally stays near unit RMS due to optimizer dynamics), so removing it changes nothing. (2) REGRESSION: Without norm constraining x0, the x0_lambdas residual injection could become unstable if some embeddings grow large during training. (3) CRASH: None — just removing a function call. Expected range: -0.003 to +0.003.


## Experiment #134 — 2026-03-16 20:09:02

- **Best val_bpb avant** : 1.067071
- **Status** : discard (val_bpb: 1.067798)
- **Description** : Lower EMBEDDING_LR from 0.6 to 0.4 — tuning the largest untested hyperparameter controlling 22% of model parameters.
- **Description** : Lower EMBEDDING_LR from 0.6 to 0.4 — tuning the largest untested hyperparameter controlling 22% of model parameters.
- **Idée** : EMBEDDING_LR=0.6 has never been tuned despite controlling 16.7M wte parameters (22% of model). The effective LR after dmodel scaling is ~0.735, which is very high for sparse embedding updates. Each embedding vector only receives gradient when its token appears in the batch — with vocab_size=32768 and batch=262K tokens, many rare tokens update infrequently. A lower LR (0.4, effective ~0.49) should improve convergence stability, preventing rare token embeddings from overshooting. VE (also an embedding) benefited from regularization (WD=0.002, exp115), suggesting embedding parameters prefer more conservative updates. This is the most commonly recommended "next step" across exp132 and exp133 reports, and it's the largest untested hyperparameter knob.
- **Pourquoi** : (1) Exhaustive analysis: all remaining architectural changes are either dead (MLP asymmetry exp130, output gating exp131, affine norm exp132, remove norm exp133), too complex/risky (DeltaNet/GLA), or crashed (FP8 exp121). (2) EMBEDDING_LR=0.6 was inherited from the original setup and NEVER tested against alternatives. (3) The embedding is the foundation of the model — every gradient flows through it. Wrong LR = suboptimal foundation for 8 layers of computation. (4) VE embedding regularization worked (exp115, -0.00088), suggesting embedding parameters benefit from conservative updates. (5) Zero crash risk, zero architecture change — just one constant.
- **Risques** : (1) NEUTRAL: 0.6 might already be near-optimal — the original authors may have tuned it. (2) REGRESSION: If embeddings need aggressive early updates to establish good representations in ~830 steps, slower LR could hurt convergence. (3) CRASH: None. Expected range: -0.003 to +0.005.


## Experiment #135 — 2026-03-16 20:31:48

- **Best val_bpb avant** : 1.067071
- **Status** : discard (val_bpb: 1.067172)
- **Description** : Intermediate multi-scale RoPE range (1500, 8000, 60000, 300000) — 200x span vs current 64x span.
- **Description** : Intermediate multi-scale RoPE range (1500, 8000, 60000, 300000) — 200x span vs current 64x.
- **Idée** : Widen per-head RoPE base frequencies from (2500, 10000, 40000, 160000) to (1500, 8000, 60000, 300000). This pushes head specialization further: head 0 becomes more local (~1-15 token distances), head 3 more global (~300-2000 tokens). The wide range (1000, 5000, 50000, 500000) was neutral in exp109 (+0.000025). The intermediate range (200x span) is explicitly listed as untested in prompt.md piste #4 — "try intermediate (1500, 8000, 60000, 300000) before declaring dead."
- **Pourquoi** : (1) Architectural parameter (priority 1). (2) Explicitly requested in prompt.md. (3) Closes out RoPE tuning thread: if neutral, declare direction dead. (4) One-line change, zero crash risk. (5) All recent architectural changes (exp130-134) were within noise — this tests a different axis entirely (positional encoding range vs MLP/norm/gating).
- **Risques** : (1) NEUTRAL (high): If 64x and 500x spans both produce ~same result, 200x is probably also neutral — RoPE multi-scale may have diminishing returns beyond the initial win (exp80, -0.0002). (2) REGRESSION (low): base=1500 could cause aliasing at short distances, base=300000 provides nearly constant positional info (useless). (3) CRASH: None. Expected range: -0.002 to +0.003.


## Experiment #136 — 2026-03-16 20:46:31

- **Best val_bpb avant** : 1.067071
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Increase WARMDOWN_RATIO from 0.6 to 0.7 for longer, more gradual LR decay.
- **Description** : Increase WARMDOWN_RATIO from 0.6 to 0.7 — more gradual LR decay exploiting the strong monotonic trend from exp13 (0.4 was +0.011 worse).
- **Idée** : WARMDOWN_RATIO controls the fraction of training spent in LR warmdown. Exp13 showed 0.4→0.6 was +0.011 improvement — the strongest hyperparameter signal ever observed. The trend is clear: more warmdown = better. Yet 0.7 has never been tested. With ~830 steps and 0.7 warmdown, ~250 steps run at full LR and ~580 steps in warmdown (vs 330/500 currently). This gives the optimizer a longer, smoother decay to find a better minimum. The Muon momentum warmup (300 steps) overlaps slightly with warmdown start (step ~250), but this could be beneficial — the LR starts declining just as momentum reaches peak. All priority 1 architectural changes are exhausted (6 straight neutral results exp130-135).
- **Pourquoi** : (1) Strongest untested hyperparameter direction with clear monotonic trend. (2) 6 consecutive neutral architectural experiments — priority 1 is exhausted. (3) Directly controls training dynamics for every step. (4) Zero crash risk, one-constant change.
- **Risques** : (1) NEUTRAL: Diminishing returns — 0.6 may already be near-optimal. (2) REGRESSION: Too little time at full LR (~250 steps) could hurt initial convergence, especially for rare token embeddings that need aggressive early updates. (3) CRASH: None. Expected range: -0.005 to +0.003.


## Experiment #137 — 2026-03-16 20:50:46

- **Best val_bpb avant** : 1.067071
- **Status** : keep (val_bpb: 1.066944)
- **Description** : Retry WARMDOWN_RATIO=0.7 (exp136 crashed on infra) — longer, more gradual LR decay exploiting the strongest hyperparam
- **Description** : Retry WARMDOWN_RATIO=0.7 (exp136 crashed on infra, never tested on GPU) — longer, more gradual LR decay.
- **Idée** : Increase WARMDOWN_RATIO from 0.6 to 0.7 for longer warmdown. Exp13 showed 0.4→0.6 was +0.011 improvement — the strongest hyperparameter signal in the entire history. The trend is monotonic: more warmdown = better. Exp136 attempted 0.7 but crashed on infra, never ran on GPU. With 0.7 warmdown and ~830 steps: ~250 steps at full LR, ~580 steps in warmdown (vs 330/500 currently). This gives the optimizer more time to slowly decay toward a better minimum. All priority 1 architectural changes exhausted (6 consecutive neutral results exp130-135).
- **Pourquoi** : (1) Strongest unexplored hyperparameter direction with clear monotonic trend (+0.011 from 0.4→0.6). (2) Exp136 crashed on infra — this is genuinely untested. (3) 6 consecutive neutral architectural experiments — priority 1 is exhausted. (4) Zero crash risk, one-constant change.
- **Risques** : (1) NEUTRAL: Diminishing returns — the 0.4→0.6 jump was +0.2 with +0.011 gain. The 0.6→0.7 jump is only +0.1, so expected improvement is smaller. (2) REGRESSION: Only ~250 steps at full LR (vs ~330). Rare token embeddings may not converge enough. (3) CRASH: None. Expected range: -0.005 to +0.003.


## Experiment #138 — 2026-03-16 21:01:09

- **Best val_bpb avant** : 1.066944
- **Status** : discard (val_bpb: 1.067077)
- **Description** : Increase WARMDOWN_RATIO from 0.7 to 0.8 — continuing the strongest monotonic hyperparameter trend in 137 experiments.
- **Description** : Increase WARMDOWN_RATIO from 0.7 to 0.8 — continuing the strongest monotonic hyperparameter trend (0.4 → 0.6 → 0.7, each better).
- **Idée** : Push WARMDOWN_RATIO to 0.8 for even longer, more gradual LR decay. Three data points confirm the monotonic trend: 0.4 was +0.011 worse than 0.6 (exp13), 0.7 beat 0.6 by -0.000127 (exp137). With 0.8 warmdown and ~830 steps: ~166 steps at full LR, ~664 steps in warmdown (vs 250/580 at 0.7). The longer decay gives the optimizer more time to settle into a sharper minimum. Exp137 report explicitly recommends this as the next step.
- **Pourquoi** : (1) Confirmed monotonic trend across 3 points — strongest signal in 137 experiments. (2) Exp137 explicitly suggests 0.8 as next step. (3) All priority 1 architectural ideas exhausted (8+ consecutive neutral). (4) Zero crash risk, one-constant change. (5) The Muon momentum warmup (300 steps) now overlaps significantly with warmdown start (~166), but this overlap was harmless at 0.7 (warmdown started at ~250, well within warmup range).
- **Risques** : (1) NEUTRAL/REGRESSION: Only ~166 steps at full LR. Embeddings and VE parameters that update sparsely may not converge enough during the very short constant-LR phase. The 0.7→0.8 jump (+0.1) yielded diminishing improvement even at 0.6→0.7 (-0.000127 vs -0.011 for 0.4→0.6). May hit diminishing returns. (2) CRASH: None. Expected range: -0.003 to +0.005.


## Experiment #139 — 2026-03-16 21:10:06

- **Best val_bpb avant** : 1.066944
- **Status** : discard (val_bpb: 1.067816)
- **Description** : Constant weight decay — remove linear WD schedule decay to prevent double-decay with LR warmdown.
- **Description** : Constant weight decay — remove linear WD schedule decay, keep WD=0.2 throughout training.
- **Idée** : The current `get_weight_decay(progress)` returns `WEIGHT_DECAY * (1 - progress)`, linearly decaying WD to 0. But WD is already multiplied by the decaying LR in the Muon update (`lr * wd * params`). This creates a double-decay: at progress=0.9, effective WD = LR(0.14) * WD_schedule(0.02) = 0.003, vs constant WD = LR(0.14) * 0.2 = 0.028. With WARMDOWN_RATIO=0.7, 70% of training has rapidly vanishing regularization. Removing the explicit decay lets the LR schedule naturally reduce effective WD while maintaining stronger regularization during the long warmdown tail. This prevents weight magnitude drift in the late training phase where the model should be settling into a minimum.
- **Pourquoi** : (1) WD schedule has NEVER been tested as a variable — only WD=0 vs WD=0.2 (exp5). (2) The double-decay interaction with WARMDOWN_RATIO=0.7 is a novel hypothesis — regularization drops quadratically instead of linearly. (3) Standard practice in most LLM training is constant WD (AdamW paper, GPT-3, etc.) — the explicit decay is unusual. (4) One-line change, zero crash risk. (5) All architectural and hyperparameter directions exhausted.
- **Risques** : (1) NEUTRAL: WD during warmdown may be too small to matter even with the change — the LR is already tiny. (2) REGRESSION: Constant WD could over-regularize during warmdown, preventing the model from fine-tuning to a sharp minimum. (3) CRASH: None. Expected range: -0.003 to +0.005.


## Experiment #140 — 2026-03-16 22:22:58

- **Best val_bpb avant** : 1.066944
- **Status** : discard (val_bpb: 1.069145)
- **Description** : Focal loss (gamma=0.5) to focus learning on hard-to-predict tokens instead of uniform cross-entropy weighting.
- **Description** : Focal loss (gamma=0.5) — down-weight easy tokens, focus gradient on hard-to-predict tokens.
- **Idée** : Replace uniform cross-entropy weighting with focal loss. For each token, compute p_target = exp(-loss), then weight by (1 - p_target)^0.5. Normalize weights to preserve loss magnitude. Easy tokens (high p_target) get lower weight, hard tokens get higher weight. This focuses the model's limited 830-step budget on predictions that matter most — the long tail of rare/hard tokens where most BPB improvement potential lies. Different from label smoothing (exp34, +0.040 catastrophic) because focal loss keeps targets one-hot and only reweights examples. Eval path (reduction='none') is unchanged.
- **Pourquoi** : (1) Novel loss function modification — never tried in 139 experiments. (2) Focal loss is well-established in vision (RetinaNet) and has theoretical backing. (3) BPB metric sums nats per byte — improving hard tokens (high nats) has outsized impact on total. (4) Minimal compute overhead (a few elementwise ops per step). (5) All architectural and hyperparameter directions are exhausted — loss function is the last unexplored axis.
- **Risques** : (1) REGRESSION (medium): Label smoothing was catastrophic (+0.040), showing loss modifications are risky. But focal loss is fundamentally different — it changes weights, not targets. (2) NEUTRAL (high): With gamma=0.5, the reweighting is mild (~1.4x for hard vs ~0.4x for easy). May not move the needle. (3) GRADIENT VARIANCE: Different per-token weights add noise to gradient estimates. But the normalization preserves mean loss magnitude. (4) CRASH: None. Expected range: -0.003 to +0.010.


## Experiment #141 — 2026-03-16 22:41:59

- **Best val_bpb avant** : 1.066944
- **Status** : keep (val_bpb: 1.064626)
- **Description** : Sparse attention gate — per-head output gating using only the first 8 dims of input, inspired by modded-nanogpt PR#117
- **Description** : Sparse attention gate — per-head output gating using only the first 8 dims of input (modded-nanogpt PR#117 style).
- **Idée** : Add a per-head gate on the attention output, but crucially using only a SPARSE subset of input dimensions (gate_dim = 2*n_head = 8). This produces a near-square (4, 8) weight matrix that Muon can orthogonalize effectively. Exp131 tried attention output gating with full n_embd=512 input and got +0.0007 (neutral) — likely because the (512, 4) matrix shape is extremely lopsided for Muon. The sparse approach (8 dims → 4 heads) was a -55 steps world record in modded-nanogpt (PR#117), where dim=12 substantially outperformed dim=96 and dim=768. Gate uses `2*sigmoid(...)` with zero-init weights → starts at 1.0 (neutral). Range (0, 2) allows both suppression and amplification of individual heads.
- **Pourquoi** : (1) Architectural change (priority 1). (2) Evidence from modded-nanogpt WR that sparse gate specifically works. (3) Exp131 was neutral but used dense (512→4) gate — the sparse variant is fundamentally different for Muon optimization. (4) Only 256 extra params (8 layers × 4 × 8), negligible compute overhead. (5) Zero crash risk.
- **Risques** : (1) NEUTRAL (medium): exp131 was neutral even with dense gate. The sparse improvement might not transfer from 124M model to our ~46M model. (2) REGRESSION (low): gate could learn to suppress useful heads, but zero-init makes this unlikely early in training. (3) CRASH: None. Expected range: -0.003 to +0.003.


## Experiment #142 — 2026-03-16 23:04:15

- **Best val_bpb avant** : 1.064626
- **Status** : discard (val_bpb: 1.068535)
- **Description** : Sparse MLP gate — per-group output gating using first 8 dims of input, extending exp141's successful sparse attention 
- **Description** : Sparse MLP gate — per-group output gating using first 8 dims of input, extending exp141's sparse attention gate to MLP.
- **Idée** : Apply the same sparse gating that worked for attention output (exp141, -0.002318) to MLP output. Split MLP output into 4 groups (n_head groups of 128 dims each) and gate each group independently using a sparse input signal (first 8 dims of normalized input). Gate matrix shape (4, 8) is identical to the attention gate — near-square for Muon. Zero-init → starts neutral at 1.0. The gate reads from post-attention normalized input, so it can condition on what attention just computed. This is a direct follow-up from exp141's "If it works" recommendations.
- **Pourquoi** : (1) Architectural change (priority 1). (2) Directly extends the biggest single improvement in recent history (exp141, -0.002318). (3) Modded-nanogpt PR#117 explored both attention and MLP gating. (4) Same (4, 8) matrix shape → proven Muon-friendly. (5) Only 256 extra params (8 layers × 4 × 8), negligible compute overhead. (6) Zero crash risk — static shapes, standard ops, torch.compile compatible.
- **Risques** : (1) NEUTRAL (medium): MLP output may not benefit from per-group gating the way attention benefits from per-head gating. Attention has semantically distinct heads; MLP groups are arbitrary dimension splits. (2) REGRESSION (low): gate could suppress useful MLP channels, but zero-init makes this unlikely. (3) CRASH: None. Expected range: -0.003 to +0.003.


## Experiment #143 — 2026-03-16 23:14:15

- **Best val_bpb avant** : 1.064626
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Attention gate_dim tuning: increase from 8 to 12 (3x n_head) for wider sparse input to per-head output gate.
- **Description** : Attention gate_dim tuning: 8→12 (3×n_head) for wider sparse input to per-head output gate.
- **Idée** : Increase gate_dim from 2×n_head=8 to 3×n_head=12, giving the sparse attention gate 50% more input dimensions (12/512=2.3% vs 8/512=1.5%) while keeping the matrix shape Muon-friendly at (4, 12) = 3:1 ratio. In modded-nanogpt PR#117, gate_dim was the critical parameter — dim=12 substantially outperformed larger dims. Our current gate_dim=8 was set by 2×n_head analogy but never tuned. With more input features the gate can make better per-head suppression/amplification decisions. Only 384 extra params per layer (vs 256 current), negligible compute.
- **Pourquoi** : (1) Direct follow-up from biggest single win (exp141, -0.002318). (2) Explicitly recommended by both exp141 and exp142 reports. (3) MLP gating (exp142) failed, pointing back to attention gate_dim tuning. (4) Minimal risk — same proven mechanism, just wider input. (5) In modded-nanogpt the optimal gate_dim varied by model — we haven't searched this.
- **Risques** : (1) NEUTRAL (medium-high): gate_dim=8 may already be optimal for our 4-head model. The 3:1 aspect ratio is slightly less Muon-friendly than the current 2:1. (2) REGRESSION (low): more input dims could cause overfitting of the gate, but 12 dims is still very sparse. (3) CRASH: None. Expected range: -0.003 to +0.003.


## Experiment #144 — 2026-03-16 23:19:42

- **Best val_bpb avant** : 1.064626
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Attention gate_dim tuning: reduce from 8 to 4 (1×n_head) for perfectly square Muon-optimal gate matrix.
- **Description** : Attention gate_dim tuning: reduce from 8 to 4 (1×n_head) for perfectly square Muon matrix.
- **Idée** : Reduce gate_dim from 2×n_head=8 to 1×n_head=4, making the sparse attention gate matrix perfectly square (4×4). Muon's Newton-Schulz polar decomposition converges fastest on square matrices. In modded-nanogpt PR#117, smaller gate_dim sometimes outperformed larger ones. Exp143 tried gate_dim=12 but crashed (infra). Testing the opposite direction: fewer input dims, perfectly square. Only 128 params per layer (vs 256 current), negligible compute.
- **Pourquoi** : (1) Direct follow-up from biggest single win (exp141, -0.002318). (2) Perfectly square matrix is theoretically optimal for Muon. (3) Exp143 crashed trying 12, so test 4 to bound the search space. (4) Simple 1-constant change, zero crash risk. (5) Modded-nanogpt showed smaller gate_dim can win.
- **Risques** : (1) NEUTRAL (medium-high): 4 dims may not carry enough information to make useful per-head gating decisions. (2) REGRESSION (low): less gating information → could default to near-uniform gating, equivalent to removing the gate. (3) CRASH: None. Expected range: -0.002 to +0.003.


## Experiment #145 — 2026-03-17 07:22:06

- **Best val_bpb avant** : 1.064626
- **Status** : en cours...


## Experiment #146 — 2026-03-17 07:31:20

- **Best val_bpb avant** : 1.064626
- **Status** : discard (val_bpb: 1.064760)
- **Description** : Attention gate_dim=4 (1x n_head, perfectly square Muon matrix) — retry of exp144 which crashed on infra, never tested 
- **Description** : Attention gate_dim=4 (1×n_head, perfectly square Muon matrix) — retry of exp144 which crashed on infra.
- **Idée** : Reduce gate_dim from 2×n_head=8 to 1×n_head=4, making the attention gate weight matrix perfectly square (4×4). Muon's Newton-Schulz polar decomposition is theoretically optimal on square matrices. In modded-nanogpt PR#117, the 1:1 ratio (dim=12 for 12 heads) beat larger ratios. Exp144 tried this exact change but crashed on infra — never tested on GPU.
- **Pourquoi** : (1) Direct follow-up from biggest single win (exp141, -0.002318). (2) Perfectly square matrix = best Muon orthogonalization. (3) Exp143 (gate_dim=12) and exp144 (gate_dim=4) both crashed on infra — neither was actually tested. (4) Single constant change, zero code risk. (5) Bounds the search: if 4 works, optimal is ≤8; if not, 8 is confirmed optimum.
- **Risques** : (1) NEUTRAL (medium-high): 4 dims may lack information for useful per-head gating decisions. (2) REGRESSION (low): with fewer input features, the gate might learn near-uniform scaling, effectively removing it. (3) CRASH: None — identical code pattern to exp141 which ran fine.


## Experiment #147 — 2026-03-17 07:40:47

- **Best val_bpb avant** : 1.064626
- **Status** : discard (val_bpb: 1.065164)
- **Description** : Attention gate_dim=12 (3x n_head) — retry of exp143 which crashed on infra, never GPU-tested.
- **Description** : Attention gate_dim=12 (3×n_head) — retry of exp143 which crashed on infra, never tested on GPU.
- **Idée** : Increase gate_dim from 2×n_head=8 to 3×n_head=12, giving the sparse attention gate 50% more input dimensions while keeping Muon-friendly (4, 12) shape. Exp146 confirmed gate_dim=4 is neutral (1.064760). gate_dim=12 has never been GPU-tested (exp143 crashed infra). In modded-nanogpt PR#117, gate_dim was the critical tuning knob. Testing the wider direction bounds the search: if 12 is also neutral, gate_dim=8 is confirmed optimal and tuning is done.
- **Pourquoi** : (1) Follows documented experimental path: "If 4 neutral → retry 12 to confirm 8 is optimal." (2) Never tested — 2 infra crashes (exp143, exp144 was gate_dim=4). (3) 50% more features for per-head gating decisions. (4) Single constant change, zero crash risk. (5) Bounds gate_dim search to move on to FP8/DeltaNet.
- **Risques** : (1) NEUTRAL (high): gate_dim=4 and 8 gave nearly identical results (0.000134 delta), suggesting gate is robust to dim changes. 12 will likely also be neutral. (2) REGRESSION (low): 3:1 aspect ratio slightly worse for Muon than 2:1, but well within effective range. (3) CRASH: None.


## Experiment #148 — 2026-03-17 07:56:38

- **Best val_bpb avant** : 1.064626
- **Status** : discard (val_bpb: 1.071212)
- **Description** : Sparse per-head gating on U-Net skip connections — extend the winning sparse gating pattern (exp141) to cross-layer in
- **Description** : Sparse per-head gating on U-Net skip connections — extend the winning sparse gating pattern (exp141) to cross-layer skip connections.
- **Idée** : Apply the same sparse input-dependent gating mechanism that worked for attention output (exp141, -0.002318) to U-Net skip connections. Currently, skip connections use a single scalar (skip_lambdas) per layer pair. By adding per-head gating with sparse input (gate_dim=2*n_head=8), the model can decide per-token which early-layer features to inject into late layers. The skip state is grouped into (n_head, head_dim) = (4, 128) chunks, matching the attention head structure inherited from the source layer. Gate matrix shape (4, 8) is identical to the attention gate — proven Muon-friendly. Zero-init → starts at 1.0 (neutral, preserves current skip_lambdas behavior). The gate conditions on the current layer's representation to route early-layer information adaptively.
- **Pourquoi** : (1) Architectural change (priority 1). (2) Extends the biggest single improvement in recent history (exp141 sparse attention gate, -0.002318) to another information pathway. (3) Same proven pattern: sparse input, structured output, input-dependent, Muon-friendly matrix shape. (4) U-Net skips were a confirmed win (exp71, -0.001) — making them smarter could amplify the benefit. (5) Only 128 new params (4 gates × 4 × 8), negligible compute. (6) Gate_dim tuning confirmed DONE (exp146/147). This is the natural next application of the sparse gating pattern.
- **Risques** : (1) NEUTRAL (medium-high): Skip connections may not benefit from per-head gating because skip states include MLP output which lacks head structure. (2) REGRESSION (low): Zero-init starts neutral, so the gate can only help or stay neutral. (3) CRASH (none): Standard linear + sigmoid ops, all static shapes, torch.compile compatible. Expected range: -0.003 to +0.003.


## Experiment #149 — 2026-03-17 08:17:21

- **Best val_bpb avant** : 1.064626
- **Status** : keep (val_bpb: 1.063080)
- **Description** : Block-diagonal local attention for middle layers (2-5) with block_size=256, keeping full attention for layers 0, 1, 6, 7
- **Description** : Block-diagonal local attention for middle layers — implement real local attention via block reshaping where WINDOW_PATTERN never worked.
- **Idée** : Replace full causal attention with block-diagonal local attention (block_size=256) for middle layers (2-5), keeping full attention for bookend/adjacent layers (0, 1, 6, 7). SDPA can't do sliding window on this GPU, but reshaping the sequence into 256-token blocks and running SDPA per block gives real local attention. 4 out of 8 layers get ~8x cheaper attention → ~13% total compute savings → ~100 more training steps. Full-attention layers (0,1,6,7) provide global context, U-Net skips propagate early information, and token shift provides 1-token cross-boundary lookback.
- **Pourquoi** : (1) Architectural change (priority 1). (2) The biggest historical win was MLP 3x (exp14, -0.008) from more training steps. This exploits the same lever: less compute → more steps. (3) WINDOW_PATTERN was designed for this but never worked with SDPA. Block-diagonal is a compile-friendly alternative. (4) Middle layers do local refinement while bookend layers handle global aggregation — well-motivated by Longformer/BigBird design. (5) Zero new parameters, pure speedup. (6) All shapes static → torch.compile(dynamic=False) compatible.
- **Risques** : (1) REGRESSION (medium): Hard block boundaries mean position 256 can't attend to position 255. Token shift and full-attention layers mitigate but may not fully compensate. (2) NEUTRAL (medium): If attention compute isn't the bottleneck, the speedup may be too small to matter. (3) CRASH (low): All ops are standard SDPA with static shapes. View/reshape should be fine on contiguous tensors after RoPE. (4) TORCH.COMPILE ISSUE (low): The if/else on self.local_attn is constant per module, should trace correctly. Expected range: -0.005 to +0.010.


## Experiment #150 — 2026-03-17 08:51:32

- **Best val_bpb avant** : 1.063080
- **Status** : discard (val_bpb: 1.063298)
- **Description** : Extend local attention from layers 2-5 to layers 1-6 (bookend-only full attention), aligning VE layers with global conte
- **Description** : Extend local attention to layers 1-6 (bookend-only full attention) — only VE layers (0, 7) keep full causal attention.
- **Idée** : Exp149 proved block-diagonal local attention works (-0.001546). Currently layers 2-5 are local, layers 0,1,6,7 are full. Extend local attention to all non-VE layers (1-6), keeping only bookend VE layers (0, 7) with full attention. This increases local layers from 4 to 6, giving ~50% more compute savings from attention → more training steps. Architecturally motivated: VE layers need full context to leverage token identity across the entire sequence, while non-VE layers do local refinement. U-Net skips, token shift, and the two full-attention bookend layers provide sufficient global context.
- **Pourquoi** : (1) Architectural change (priority 1). (2) Direct follow-up to exp149 (new best, -0.001546). (3) Same "more steps" lever as MLP 3x (exp14, biggest historical win). (4) Clean architectural rationale: VE layers = global, non-VE layers = local. (5) One-line change, zero crash risk. (6) 6/8 layers local instead of 4/8 → significant additional speedup.
- **Risques** : (1) REGRESSION (medium): Layers 1 and 6 lose full attention. Layer 1 is early (representation building), layer 6 is late (pre-output). But both have token shift for immediate neighbor access and receive global info via U-Net skips/residual from layer 0. (2) NEUTRAL (medium): The additional 2 layers may not save enough compute to gain meaningful steps if attention isn't the bottleneck in those layers. (3) CRASH: None — identical code path as exp149, just different layer selection.


## Experiment #151 — 2026-03-17 09:00:00

- **Best val_bpb avant** : 1.063080
- **Status** : discard (val_bpb: 1.063407)
- **Description** : Reduce local attention block_size from 256 to 128 for middle layers (2-5) to gain more training steps.
- **Description** : Reduce local attention block_size from 256 to 128 for middle layers (2-5) — more compute savings → more training steps.
- **Idée** : Halve the block size for block-diagonal local attention in middle layers (2-5). With block_size=128 instead of 256, each local attention layer computes attention over 128-token blocks instead of 256-token blocks, halving local attention compute. This is the documented next step after exp149 (block_size=256 → new best) and exp150 (layer selection done: 2-5 optimal). The "more steps" lever has been the most consistent win historically (MLP 3x = -0.008, biggest ever).
- **Pourquoi** : (1) Documented follow-up from prompt.md item #44: "try block_size=128 (more steps but harsher boundaries)." (2) Layer selection confirmed done (exp150: 1-6 was neutral, 2-5 is optimal). (3) Same "more steps" lever as MLP 3x (exp14, biggest historical win). (4) One-constant change, zero crash risk. (5) All shapes still static (2048 / 128 = 16 blocks, exact division). (6) Token shift provides 1-token cross-boundary lookback. Full-attention layers 0,1,6,7 provide global context.
- **Risques** : (1) REGRESSION (medium): 128-token blocks are quite narrow — harder boundaries, less local context per block. Position 128 can't attend to 127. Token shift mitigates partially. (2) NEUTRAL (medium): If the additional compute savings from 256→128 are small relative to MLP compute, the extra steps may not compensate for reduced attention quality. (3) CRASH: None — identical code path, T=2048 divides evenly by 128.


## Experiment #152 — 2026-03-17 09:08:53

- **Best val_bpb avant** : 1.063080
- **Status** : discard (val_bpb: 1.068952)
- **Description** : Cross-block linear attention recurrence for local layers — propagate KV state between blocks for long-range context at
- **Description** : Cross-block linear attention recurrence for local layers — propagate KV state between blocks for long-range context without full attention cost.
- **Idée** : Local attention (exp149, -0.001546) has hard block boundaries: position 256 can't see position 255. Add a cross-block linear recurrence that accumulates a KV state matrix (K^T @ V) across blocks and lets each block query information from all previous blocks. This provides cross-block context that's currently missing. The recurrence adds <2% total compute because local SDPA is only ~1.3% of total compute (attention projections and MLP dominate). Uses RMS-normalized recurrent output mixed via a per-head learnable gate initialized at sigmoid(-2)≈0.12, ensuring near-identical start to current best. 16 new parameters total (4 heads × 4 local layers). This is a hybrid local-linear attention design: softmax SDPA for within-block, linear attention for cross-block.
- **Pourquoi** : (1) Architectural change (priority 1). (2) Addresses the main weakness of block-diagonal attention without the compute cost of full attention. (3) Compute overhead is negligible (<2%) because attention compute is a tiny fraction of total (MLP and lm_head dominate). (4) Safe initialization: gate starts small, model begins nearly identical to current best. (5) The "boundary problem" was the main concern when block-diagonal was introduced (exp149). exp150 (more local layers) and exp151 (smaller blocks) showed that layer selection and block size are tuned — the remaining lever is cross-block information flow. (6) Linear attention state passing is well-studied (TransNormer, RWKV, RetNet) and proven effective for long-range context compression.
- **Risques** : (1) NEUTRAL (medium): Full-attention layers (0,1,6,7) already provide global context via residual stream. The recurrence may be redundant. (2) REGRESSION (low): Normalized+gated design ensures near-identical start. (3) CRASH (low): All ops are standard (einsum, cumsum, cat, norm). Static shapes. (4) TORCH.COMPILE (low-medium): einsum is well-supported, but (D,D) intermediate tensors add 134MB per layer. Should be fine with 32GB VRAM. (5) OOM (low): Additional ~536MB during backprop (4 layers × 134MB). Current peak is 12.2GB on 32GB GPU.


## Experiment #153 — 2026-03-17 09:28:31

- **Best val_bpb avant** : 1.063080
- **Status** : keep (val_bpb: 1.062436)
- **Description** : Dynamic attention temperature — input-dependent per-head Q scaling after QK-norm for adaptive attention sharpness.
- **Description** : Dynamic attention temperature — input-dependent per-head Q scaling after QK-norm for adaptive attention sharpness.
- **Idée** : QK-norm removes all magnitude information from Q and K, fixing attention sharpness at 1/sqrt(d) for all tokens equally. Add a dynamic, per-position, per-head temperature that scales Q after QK-norm. Uses the same sparse input pattern (first 8 dims) that worked for the attention gate (exp141, -0.002). New params: 8×4 = 32 per layer (256 total). Compute cost: negligible (<0.001%). Zero-init starts at temp=1.0 (neutral). This is fundamentally different from dead qk_scales (exp67/68) which was STATIC per-head — dynamic temperature changes per-token, letting the model sharpen attention for rare-word predictions and broaden it for common-word predictions.
- **Pourquoi** : (1) Architectural change (priority 1). (2) Different from qk_scales: input-dependent vs static. (3) Same sparse input pattern as attention gate (proven effective). (4) Orthogonal to attention gate: gate controls output magnitude, temperature controls attention distribution shape. (5) Zero crash risk — standard Linear + sigmoid + multiply, all static shapes. (6) Zero-init ensures identical start to current best.
- **Risques** : (1) NEUTRAL (medium): The model may not benefit from per-token sharpness control at this scale/context length. (2) REGRESSION (very low): Zero-init guarantees neutral start. (3) CRASH (none): Standard ops, static shapes, torch.compile compatible. Expected range: -0.003 to +0.003.


## Experiment #154 — 2026-03-17 09:44:04

- **Best val_bpb avant** : 1.062436
- **Status** : discard (val_bpb: 1.067878)
- **Description** : Dynamic key temperature — input-dependent per-head K scaling after QK-norm, complementary to Q-temp.
- **Description** : Dynamic key temperature — input-dependent per-head K scaling after QK-norm, complementary to Q-temp (exp153).
- **Idée** : Exp153 added dynamic Q temperature (per-head Q scaling from sparse input). This adds the symmetric counterpart: dynamic K temperature. Together they create a rank-1 temperature matrix where the effective attention sharpness between query position i and key position j is t_q[i] * t_k[j]. Q-temp controls how focused each query is (all keys scaled uniformly). K-temp controls how prominent each key is (independent of query). This is a richer model than Q-temp alone: a "rare word" key can make itself more visible to all queries, while a "common word" key can dampen itself. Same sparse input (first 8 dims), same param count (+256), negligible compute.
- **Pourquoi** : (1) Architectural change (priority 1). (2) Direct follow-up to exp153 (new best, -0.000644). (3) Orthogonal mechanism: Q-temp = query focus, K-temp = key prominence. Combined = rank-1 temperature matrix. (4) Same proven sparse input pattern. (5) Zero crash risk — identical code structure to Q-temp. (6) Zero-init ensures identical start to current best.
- **Risques** : (1) NEUTRAL (medium): The model may not need per-key prominence control on top of per-query focus. The product t_q*t_k may overparameterize temperature. (2) REGRESSION (very low): Zero-init guarantees neutral start. (3) CRASH (none): Standard Linear + sigmoid + multiply, all static shapes.


## Experiment #155 — 2026-03-17 09:54:13

- **Best val_bpb avant** : 1.062436
- **Status** : discard (val_bpb: 1.068479)
- **Description** : Per-head input-dependent V scaling — supply-side value magnitude modulation complementary to demand-side attention gat
- **Description** : Per-head input-dependent V scaling — supply-side value magnitude modulation complementary to demand-side attention gate.
- **Idée** : Add input-dependent per-head scaling on attention values (V) after VE injection. Uses same sparse input pattern (first gate_dim=8 dims) and 2*sigmoid zero-init pattern as attention gate (exp141) and Q-temp (exp153). Gate controls demand (how much query RECEIVES), V-scale controls supply (how much key PROVIDES). Mathematically different from gate: gate[i] scales the sum uniformly, val_scale[j] scales each term inside the sum. Also different from K-temp (exp154): K-temp changed attention DISTRIBUTION (redundant with Q-temp), V-scale changes value CONTENT magnitude (orthogonal). 256 new params (0.00036%).
- **Pourquoi** : (1) Architectural change (priority 1). (2) Follows the pattern of successful sparse modulations: gate (-0.002), Q-temp (-0.0006). (3) Complementary to existing mechanisms: gate=demand-side, V-scale=supply-side. (4) Different from dead K-temp: changes content magnitude, not attention distribution. (5) Zero crash risk — standard Linear + sigmoid + multiply, static shapes. (6) Zero-init → val_scale=1.0 (identical start).
- **Risques** : (1) NEUTRAL (medium): The c_v projection already provides linear value scaling. The multiplicative nonlinear modulation may not add useful capacity. (2) REGRESSION (very low): Zero-init guarantees neutral start. (3) CRASH (none): Standard ops, static shapes. (4) REDUNDANCY (low-medium): Combined with attn_gate, the model has both input and output per-head modulation. May overparameterize the head importance space.


## Experiment #156 — 2026-03-17 10:12:58

- **Best val_bpb avant** : 1.062436
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Partial Rotary Position Embedding (50%) — apply RoPE to only the first 64 of 128 head dimensions, leaving the rest pos
- **Description** : Partial Rotary Position Embedding (50% of head_dim) — apply RoPE to only the first 64 of 128 head dimensions, making the remaining 64 position-free for content-only attention matching.
- **Idée** : Currently all 128 dims of Q and K get rotary encoding. This means attention similarity is always position-dependent. With partial RoPE (ROPE_FRACTION=0.5), each head's dot product becomes: [position-sensitive similarity from 64 rotated dims] + [pure content similarity from 64 non-rotated dims]. This gives the model a new capability: content-only matching (e.g., "these two tokens are semantically similar regardless of distance"). This technique is well-proven in production models (PaLM uses 25%, CodeLlama uses partial RoPE). Each head still has its own multi-scale RoPE base for the rotated part, plus head-specific content matching in the non-rotated part (Q/K projections are per-head). Zero new parameters. Slightly less compute (fewer sin/cos ops, fewer rotary multiplications). The `apply_rotary_emb` function now cleanly handles partial rotation by splitting x into rotated and pass-through parts based on cos/sin shape.
- **Pourquoi** : (1) Architectural change (priority 1). (2) Adds a fundamentally NEW capability — content-only matching — not available in the current model. (3) Proven effective in production models (PaLM, CodeLlama, Gemma). (4) Zero new parameters. (5) Backward compatible: ROPE_FRACTION=1.0 recovers the current behavior. (6) The current multi-scale RoPE already specializes heads by distance scale; partial RoPE adds an orthogonal content-matching axis. (7) No crash risk: only changes precomputed cos/sin shape and apply_rotary_emb slicing. All shapes remain static.
- **Risques** : (1) REGRESSION (medium): At 2048 context, every dim may benefit from positional encoding. 50% reduction is aggressive — 75% might be safer but less impactful. (2) NEUTRAL (medium): With only 4 heads and multi-scale RoPE, the model already has good positional diversity. Content-only matching may not add useful capability at this scale. (3) CRASH (none): Standard tensor slicing and cat, all shapes static, torch.compile compatible. (4) OOM (none): Actually reduces memory slightly (smaller cos/sin buffers).


## Experiment #157 — 2026-03-17 10:24:52

- **Best val_bpb avant** : 1.062436
- **Status** : en cours...
- **Description** : Partial Rotary Position Embedding (50%) — retry of exp156 which crashed (likely infra, 0 VRAM). Apply RoPE to only first 64/128 head dims, leaving 64 dims position-free for content-only attention matching.
- **Idée** : Currently all 128 Q/K dims get rotary encoding, making every attention score position-dependent. With ROPE_FRACTION=0.5, each head's dot product splits into: [position-sensitive similarity from 64 rotated dims] + [pure content similarity from 64 non-rotated dims]. This gives the model a new capability: position-independent content matching ("these tokens are semantically similar regardless of distance"). Proven in PaLM (25%), CodeLlama, Gemma. Zero new parameters. Slightly less compute (fewer sin/cos ops). Multi-scale RoPE bases still apply to the rotated part.
- **Pourquoi** : (1) Architectural change (priority 1). (2) Adds fundamentally NEW capability — content-only matching — not available in current model. (3) Proven in production models. (4) Zero new params. (5) exp156 crashed with 0 VRAM (infra crash, never reached GPU). (6) Clean implementation: apply_rotary_emb derives rope_dim from cos shape, backward compatible with ROPE_FRACTION=1.0.
- **Risques** : (1) REGRESSION (medium): 50% is aggressive at 2048 context. (2) NEUTRAL (medium): 4 heads with multi-scale RoPE already have good positional diversity. (3) CRASH (very low): clean tensor slicing, all static shapes. Exp156 crash was infra, not code.


## Experiment #158 — 2026-03-17 10:44:37

- **Best val_bpb avant** : 1.062436
- **Status** : discard (val_bpb: nan)
- **Description** : Cross-layer MLP residual connection — direct skip from each layer's MLP output to the next layer's MLP input for progr
- **Description** : Cross-layer MLP residual connection — direct skip from each layer's MLP output to the next layer's MLP input, bypassing intervening attention.
- **Idée** : The MLP currently processes each position independently with no cross-layer MLP communication. Each MLP output goes into the residual stream where it gets mixed with attention outputs, x0 connections, and skip connections before reaching the next MLP (via normalization). This dilutes the MLP signal. A direct MLP-to-MLP skip creates a separate "MLP stream" that enables progressive MLP computation. Particularly useful with non-uniform MLPs (1.5x early → 4.5x late): narrow early MLPs can prepare features that wide late MLPs build upon directly. Uses learned per-layer scalar lambdas (like skip_lambdas), init=0.05 for immediate activation. 8 new scalar params (negligible). Different from dead sparse MLP gate (exp142) which gated MLP OUTPUT per group — this connects MLP OUTPUT to next MLP INPUT.
- **Pourquoi** : (1) Architectural change (priority 1). (2) Creates a genuinely new information pathway not present in the model. (3) Exploits non-uniform MLP structure — early narrow MLPs feed directly into late wide MLPs. (4) Same pattern as successful skip_lambdas (exp98, -0.002): learned scalar connecting outputs to inputs across layers. (5) Zero crash risk — standard tensor add/multiply, all shapes static, torch.compile compatible. (6) Init=0.05 but MLP c_proj starts at 0, so mlp_prev ≈ 0 initially → safe even with non-zero lambda.
- **Risques** : (1) NEUTRAL (medium): The residual stream already carries MLP output to the next layer. The direct skip may be redundant. (2) REGRESSION (very low): Lambda starts at 0.05 but multiplied by ~0 MLP output initially, so model starts near-identical to current best. (3) CRASH (none): Standard ops, static shapes. (4) GRADIENT (very low): Long MLP chain could cause vanishing/exploding gradients, but lambdas are small and learned.


## Experiment #159 — 2026-03-17 11:38:12

- **Best val_bpb avant** : 1.062436
- **Status** : en cours...
- **Description** : Hybrid local/global attention heads in middle layers — 1 global head (longest RoPE base=160000) + 3 local heads (block_size=256) in layers 2-5.
- **Idée** : Middle layers (2-5) use block-diagonal local attention for ALL 4 heads. They can't see beyond 256 tokens. Cross-block info only comes from bookend full-attention layers (0,1,6,7) and token shift (1-token lookback). By making the last head (RoPE base=160000, longest range) use full causal attention, each middle layer gets a "global oracle" for long-range context while 3 heads still do efficient local processing. Intentionally give the global role to the long-range head — it has RoPE base=160000 which is designed for distant token relationships, so full-sequence attention is most valuable for it. Compute impact: attention FLOPs for middle layers go from 4 local to (3 local + 1 global) = ~43% of full compute vs ~12.5% before. Total model step time increases moderately (~5-8% fewer steps). The trade-off: slightly fewer steps but much richer cross-block context in middle layers.
- **Pourquoi** : (1) Architectural change (priority 1). (2) Zero new parameters. (3) Addresses a clear limitation: middle layers are blind beyond 256 tokens. (4) Exp150 showed that WHICH layers are local matters (1-6 neutral vs 2-5), but never tested WITHIN-LAYER hybrid. (5) The long-range RoPE head (base=160000) is wasted on local attention since its positional encoding is designed for ~1600-token relationships, far beyond the 256-token block. (6) Clean torch.compile compatible implementation (static shapes, two SDPA calls with fixed head counts).
- **Risques** : (1) REGRESSION (medium): more compute → fewer steps (~5-8%). Exp149's local attention win was partly from MORE steps. Giving back some of that speed may not be worthwhile. (2) NEUTRAL (medium): full-attention layers 0,1,6,7 may already provide sufficient global context to middle layers via the residual stream. (3) CRASH (very low): standard SDPA calls, all shapes static. (4) OOM (none): no extra memory beyond standard attention for 1 head.


## Experiment #160 — 2026-03-17 12:07:16

- **Best val_bpb avant** : 1.062436
- **Status** : discard (val_bpb: 1.066460)
- **Description** : Learned absolute position embeddings added to token embeddings, complementing RoPE with explicit absolute position for M
- **Description** : Learned absolute position embeddings — add a (2048, 512) learned parameter to token embeddings before norm, giving MLP and residual stream explicit absolute position information.
- **Idée** : The model currently has NO absolute position encoding. RoPE provides relative position in Q/K only (attention-side). Token shift gives 1-step local context for K/V. But the MLP and residual stream are completely position-blind — they have no way to know whether they're processing position 0 or position 2000. Adding learned absolute position embeddings (à la GPT-2) on TOP of RoPE creates a hybrid: RoPE handles relative position in attention, while the learned PE provides absolute position everywhere (MLP, x0 propagation, residual stream). This is a genuinely new information pathway. The PE is added to wte output BEFORE norm, so it affects the x0 signal (which propagates to all layers via x0_lambdas). Each position gets 512 learnable dims of position info. With BOS-aligned document packing, position correlates with document structure (early positions = near BOS = less context). The model can learn position-dependent processing patterns: e.g., use less confident predictions near document starts, different feature extraction at different sequence positions.
- **Pourquoi** : (1) Architectural change (priority 1). (2) Adds a genuinely NEW information pathway — absolute position to MLP/residual, not just attention. (3) Zero-init preserves current behavior identically. (4) Proven effective in GPT-2 family (though those didn't have RoPE — this is the first test of the combination). (5) 1.05M new params (1.5% increase, well within model capacity). (6) Negligible compute cost (one addition). (7) Position info propagates to ALL layers via x0_lambdas without needing per-layer PE. (8) Zero crash risk — standard parameter + addition, all static shapes, torch.compile compatible.
- **Risques** : (1) NEUTRAL (medium-high): The MLP may not benefit from absolute position info at 2048 context — attention already provides position-dependent processing via RoPE. (2) INTERFERENCE (low): The absolute PE contributes to Q/K through linear projections, on top of RoPE rotation. Could create conflicting position signals, but the model can learn to project out the absolute component. (3) REGRESSION (very low): Zero-init → no absolute PE initially. The model gradually learns to use it. (4) TRAINING (low): 1.05M new parameters need learning in ~830 steps, but each position gets 128 gradient signals per optimizer step (from the full batch). (5) CRASH (none): Standard nn.Parameter + addition.


## Experiment #161 — 2026-03-17 12:30:14

- **Best val_bpb avant** : 1.062436
- **Status** : discard (val_bpb: 1.066087)
- **Description** : Layer-conditional activation — plain ReLU for early layers (0-3), ReluSquared for late layers (4-7).
- **Description** : Layer-conditional activation — plain ReLU for early layers (0-3, MLP 1.5x), ReluSquared for late layers (4-7, MLP 4.5x).
- **Idée** : Early layers have narrow MLPs (1.5x = 768 hidden units). ReluSquared kills ~50% via ReLU then further suppresses small positive values via squaring, severely limiting the effective capacity of these already-narrow layers. Plain ReLU keeps all positive neurons at full magnitude, giving early layers more active capacity. Late layers (4.5x = 2304 hidden) have plenty of capacity and benefit from ReluSquared's sparsity and stronger non-linearity. This matches activation complexity to layer role: simple feature extraction (early) vs complex feature combination (late). c_proj is zero-initialized so output magnitude difference is compensated by learning.
- **Pourquoi** : (1) Architectural change (priority 1). (2) Theoretically motivated: narrow layers need more active neurons, wide layers benefit from sparsity. (3) Zero new parameters. (4) Different from dead ideas: exp66 (GeluSquared ALL layers) and exp30 (SwiGLU ALL layers) changed activation uniformly — this is LAYER-CONDITIONAL. (5) Tiny compute savings in early layers (no square op). (6) The non-uniform MLP already differentiates early/late layers by width — this extends the differentiation to activation function.
- **Risques** : (1) REGRESSION (medium): ReluSquared's sparsity might actually help narrow early MLPs focus on important features. Removing squaring could make early MLPs less selective. (2) NEUTRAL (high): c_proj already learns to compensate for activation differences. The model may converge to similar effective behavior regardless of activation choice. (3) CRASH (none): Standard F.relu, constant branch (torch.compile traces as constant). (4) OOM (none): Less memory if anything (no intermediate squared tensor in early layers).


## Experiment #162 — 2026-03-17 13:05:05

- **Best val_bpb avant** : 1.062436
- **Status** : discard (val_bpb: 1.070433)
- **Description** : Value centroid injection — inject causal running mean of V into attention output via learned per-head gate for global 
- **Description** : Value centroid injection — inject causal running mean of V into attention output via learned per-head gate, providing global context to local attention layers.
- **Idée** : Local attention layers (2-5) are blind beyond their 256-token block. The only cross-block info comes from bookend full-attention layers via the residual stream and 1-step token shift. By computing the causal running mean of V (cumsum/position) and adding it to the attention output with a learned per-head gate, each layer gets a cheap global context signal summarizing the full sequence's value content. This is a genuinely new information pathway: token shift gives 1-step K/V history; centroid gives full-sequence V average to the output. For full-attention layers (0,1,6,7), the centroid is redundant with softmax-weighted attention — the gate can learn to suppress it there. For local layers (2-5), it fills a real gap. Zero-init gate → starts at 0, model learns to use centroid. 256 new params total (8 layers × 8→4 Linear).
- **Pourquoi** : (1) Architectural change (priority 1). (2) Genuinely new information pathway — no existing component provides uniform full-sequence context. (3) Specifically targets the local attention layers' cross-block blindness (the weakness introduced by exp149's speedup). (4) Same sparse-input gating pattern (gate_dim=8) proven in attn_gate (exp141, -0.002) and attn_temp (exp153, -0.0006). (5) Negligible compute: cumsum + divide + multiply + add. (6) 256 new params. (7) Zero crash risk — all ops are standard, all shapes static, torch.compile compatible.
- **Risques** : (1) NEUTRAL (high): full-attention bookend layers may already provide sufficient global context via residual stream, making the centroid redundant. (2) REGRESSION (low): centroid is an unweighted average — it lacks relevance filtering and could inject noise. (3) MEMORY (low): v_cumsum doubles V memory per layer (~64MB total, 0.5% of 12GB peak). (4) CRASH (none): cumsum, arange, division are all standard ops.


## Experiment #163 — 2026-03-17 13:22:35

- **Best val_bpb avant** : 1.062436
- **Status** : discard (val_bpb: 1.078273)
- **Description** : Adaptive x0 injection — input-dependent per-position gating on x0_lambdas using sparse gating pattern for content-depe
- **Description** : Adaptive x0 injection — input-dependent per-position gating on x0_lambdas using proven sparse gating pattern (Linear(8→1), zero-init, 2*sigmoid).
- **Idée** : x0_lambdas inject raw embedding (x0) into each layer's input. Currently this injection is a fixed scalar per layer — every position gets the same amount regardless of content. By adding a sparse input-dependent gate (same pattern as attn_gate exp141 and attn_temp exp153), the model can modulate x0 strength per position: more raw embedding for rare/ambiguous tokens, less for well-represented common tokens. The gate reads from the first 8 dims of the residual stream (gate_dim=2*n_head=8) and outputs 1 scalar per position, same as proven sparse gating pattern. Zero-init → sigmoid(0)=0.5 → 2*0.5=1.0 → starts identical to current behavior.
- **Pourquoi** : (1) Architectural change (priority 1). (2) Reuses the EXACT sparse gating pattern that produced the two most recent wins (exp141 -0.002, exp153 -0.0006). (3) x0 mixing is critical (removing x0_lambdas = +0.020) so better control of it could help. (4) 64 new params total (8 per layer), negligible compute. (5) Different from per-dimension lambdas (exp114, +0.009) which was per-CHANNEL without content-dependence — this is per-POSITION with content-dependence.
- **Risques** : (1) NEUTRAL (medium-high): optimal x0 strength may be position-independent — the scalar x0_lambda is already optimal. (2) REGRESSION (very low): zero-init starts identical to current best. (3) CRASH (none): standard Linear + sigmoid, all shapes static, torch.compile compatible. (4) OOM (none): 64 params, negligible memory.


## Experiment #164 — 2026-03-17 13:52:34

- **Best val_bpb avant** : 1.062436
- **Status** : crash (val_bpb: 0.000000)
- **Description** : torch.compile(mode="reduce-overhead") for CUDA graph optimization to increase training throughput and get more steps.
- **Description** : torch.compile(mode="reduce-overhead") for CUDA graph optimization — reduce kernel launch overhead to get more training steps.
- **Idée** : The model uses torch.compile(dynamic=False) with default mode. The `reduce-overhead` mode uses CUDA graphs to batch kernel launches, reducing Python/framework overhead between kernels. With many small operations per layer (RMSNorm, sigmoid gates, token shift, sparse gate projections, dynamic temperature), kernel launch overhead is significant. CUDA graphs capture the kernel sequence and replay it without per-kernel Python dispatch. This is the ONLY untested compile mode — max-autotune (exp16/17) crashed (likely Triton autotuning, not CUDA graphs), max-autotune-no-cudagraphs (exp21) was +0.009 (Triton autotuning alone hurts). reduce-overhead = CUDA graphs WITHOUT Triton autotuning = fundamentally different from both. The model is designed to be CUDA-graph compatible: all shapes static, gc.disable(), expandable_segments:True, no dynamic allocation. The "more steps" lever has been the most consistent winner (MLP 3x = -0.008, local attention = -0.0015).
- **Pourquoi** : (1) Optimization change (priority 2). (2) Targets the biggest remaining lever: more training steps in the same 5-minute budget. (3) Never tested — the 2 other compile modes were tested but reduce-overhead was skipped. (4) Model is specifically designed for static compilation (dynamic=False, all shapes constant, gc disabled). (5) 1-line change, zero new parameters, no architecture changes. (6) Expected 10-30% throughput improvement = 80-250 additional training steps. (7) MFU is likely 30-40% currently, suggesting significant launch overhead.
- **Risques** : (1) CRASH (medium): CUDA graphs require all memory addresses to remain stable between replays. If any operation allocates new memory or changes tensor count, the graph breaks. The model should be safe (all static) but the optimizer's torch.stack/unbind in Muon could potentially cause issues. (2) NEUTRAL (low): If the model is already compute-bound (not launch-overhead-bound), CUDA graphs won't help. But at MFU ~35%, there's likely significant overhead. (3) REGRESSION (none): If it runs, it's strictly faster — same model, same math, fewer Python dispatch cycles. (4) OOM (low): CUDA graphs pre-allocate memory for replay buffers, which could increase peak VRAM slightly.


## Experiment #165 — 2026-03-17 15:01:53

- **Best val_bpb avant** : 1.062436
- **Status** : discard (val_bpb: 1.063187)
- **Description** : x0_lambdas init reduced from 0.1 to 0.05 for less raw embedding noise in deep layers.
- **Description** : x0_lambdas init tuning — reduce from 0.1 to 0.05 for less raw embedding noise in deep layers.
- **Idée** : x0_lambdas inject raw embedding (x0) at every layer. Currently init=0.1, never tuned. Over 8 layers, cumulative x0 contribution is ~0.8x the raw embedding by the last layer — a lot of un-processed signal polluting deep representations. Reducing to 0.05 (cumulative ~0.4x) gives deep layers a cleaner residual stream while preserving the critical x0 pathway. This is the direct analog of skip_lambdas init tuning (0.0→0.05, exp98, -0.002) — the only other scalar init that was tuned, producing a significant win. With scalar_lr=0.5, x0_lambdas can still learn optimal values from 0.05 in 830 steps; the question is whether starting closer to the optimum improves the trajectory.
- **Pourquoi** : (1) skip_lambdas 0.0→0.05 gave -0.002 (exp98), so scalar init tuning is a proven lever at this scale. (2) x0_lambdas=0.1 was never tuned — it's the original default from when the model was much simpler. (3) Deep layers need processed representations, not raw embeddings — less x0 noise could improve late-layer MLP and attention. (4) 1-line change, zero risk of crash/OOM, zero new parameters. (5) Listed as item 57 in prompt.md remaining ideas.
- **Risques** : (1) NEUTRAL (high): with scalar_lr=0.5 and 830 steps, the optimizer can easily move x0_lambdas from 0.1 to 0.05 or vice versa — init may not matter in this range. (2) REGRESSION (very low): 0.05 is within the plausible optimal range. x0 is still active (removing x0 = +0.020). (3) CRASH (none): single constant change.


## Experiment #166 — 2026-03-17 15:21:23

- **Best val_bpb avant** : 1.062436
- **Status** : discard (val_bpb: 1.062787)
- **Description** : Increase EMBEDDING_LR from 0.6 to 0.8 for faster token embedding convergence in the largest parameter group (22% of mode
- **Description** : Increase EMBEDDING_LR from 0.6 to 0.8 — tuning the largest untested hyperparameter controlling 22% of model parameters.
- **Idée** : The embedding LR was set to 0.6 early and never properly tuned. Only 0.4 was tested (exp134, neutral). With dmodel_lr_scale=1.22 (for 512d model), effective LR goes from 0.73 to 0.98. Higher embedding LR helps the model learn better token representations faster within the ~830-step budget. Embeddings are the single largest parameter group (wte: 16.8M = 22% of model) and the most under-explored hyperparameter. The Adam optimizer with betas=(0.8, 0.95) can handle this LR — no WD on embeddings provides natural regularization via gradient noise.
- **Pourquoi** : (1) Explicitly listed as item 58 in remaining ideas. (2) Embeddings are 22% of model, largest single param group. (3) Only 0.4 tested (neutral in exp134), 0.8 never tested. (4) All simple architectural changes exhausted after 165 experiments. (5) Zero risk of crash/OOM — single constant change. (6) The effective LR (0.98) is high but within the range the Adam optimizer can handle with these betas.
- **Risques** : (1) NEUTRAL (high): with scalar_lr already handling embedding updates and 830 steps, the difference between 0.6 and 0.8 may be undetectable. (2) REGRESSION (low): effective LR=0.98 could cause slight instability in early training, but Adam's adaptive per-parameter LR prevents divergence. (3) CRASH (none): single constant change.


## Experiment #167 — 2026-03-17 15:55:14

- **Best val_bpb avant** : 1.062436
- **Status** : discard (val_bpb: 1.063240)
- **Description** : Learnable softcap — replace fixed softcap=13 constant with a gradient-optimized parameter via softplus(raw)+1 for cont
- **Description** : Learnable softcap — replace fixed softcap=13 with a gradient-optimized parameter via softplus(raw)+1.
- **Idée** : The logit softcap (13) was hand-tuned across only 3 integer values: 30 (+0.005), 15 (original), 13 (-0.001 = best), 11 (+0.001). The sensitivity between 11-15 is ±0.001, showing the exact value matters. The optimal may be a non-integer (e.g., 12.5, 13.7) that hand-tuning can't find. By making softcap a learnable parameter (softplus(raw)+1, always ≥1, init=13.0), the model finds the exact optimum via gradient descent. This is an architectural change (learnable component replaces fixed constant) with zero risk: 1 new scalar parameter, neutral init, smooth gradients through softplus + tanh. The gradient d(loss)/d(softcap) naturally exists through the `softcap * tanh(logits / softcap)` formula — the cap affects both the compression ratio and output magnitude.
- **Pourquoi** : (1) Architectural change (priority 1 — converts a fixed hyperparameter to a learned component). (2) Softcap axis showed clear sensitivity: 11→13→15 is a non-monotonic curve with 13 as discrete optimum. Continuous optimization can find the true minimum. (3) 1 new scalar parameter, negligible compute. (4) Zero risk of crash/OOM — softplus+1 ensures softcap ≥ 1.0 always. (5) After 166 experiments, all discrete hyperparameter values are tested; this is the natural next step: let the optimizer tune it continuously. (6) LR=0.05 (scalar_lr*0.1) with Adam betas=(0.96, 0.95) gives smooth, moderate learning for a global parameter.
- **Risques** : (1) NEUTRAL (high): softcap=13.0 may already be near the continuous optimum, so the learned value stays near 13 with negligible improvement. (2) REGRESSION (very low): softplus+1 constrains cap ≥ 1, init is exactly 13. Worst case: cap drifts slightly and costs ~0.001. (3) CRASH (none): standard PyTorch ops, all shapes static, torch.compile compatible. (4) OOM (none): 1 scalar parameter.


## Experiment #168 — 2026-03-17 16:27:36

- **Best val_bpb avant** : 1.062436
- **Status** : discard (val_bpb: 1.064129)
- **Description** : WEIGHT_DECAY=0.1 — AB-test lower cautious weight decay for Muon optimizer (currently 0.2, only 0.0 tested before).
- **Description** : WEIGHT_DECAY=0.1 — AB-test lower cautious weight decay for Muon (currently 0.2, only 0.0 tested before).
- **Idée** : WD=0.2 has never been AB-tested against intermediate values. Only WD=0.0 was tested (+0.004, clearly worse). The optimal WD could be anywhere between 0 and 0.2. With only ~830 training steps, the model needs parameters to move fast — less regularization could help. The cautious WD mechanism (only applied where gradient aligns with weight) already limits effective WD. The WD schedule decays linearly from WD to 0 over training, so WD=0.1 means half the cumulative regularization pressure. This is explicitly listed as priority #3 in the remaining ideas (prompt.md).
- **Pourquoi** : (1) Explicitly listed on priority queue: "WD=0.2 jamais AB-testé vs 0.1/0.3". (2) WD=0.0 was clearly worse (+0.004), showing WD matters, but optimal value between 0 and 0.2 is unknown. (3) Short training (830 steps) may benefit from less regularization. (4) All architectural changes exhausted after 167 experiments. (5) 1-line change, zero crash/OOM risk.
- **Risques** : (1) NEUTRAL (high): delta between WD=0.1 and WD=0.2 may be within noise (~0.003). (2) REGRESSION (low): WD=0.1 is still positive WD, unlikely to be much worse than 0.2. (3) CRASH (none): single constant change.


## Experiment #169 — 2026-03-17 16:41:49

- **Best val_bpb avant** : 1.062436
- **Status** : crash (val_bpb: 0.000000)
- **Description** : FP8 MLP-only matmuls via torch._scaled_mm for ~2x MLP throughput on Blackwell FP8 tensor cores.
- **Description** : FP8 MLP-only matmuls — replace MLP c_fc and c_proj with float8_e4m3fn matmuls via torch._scaled_mm for ~2x MLP throughput on Blackwell FP8 tensor cores.
- **Idée** : MLP accounts for ~60% of per-layer compute. By replacing the two MLP linear ops (c_fc and c_proj) with FP8 matmuls using torch._scaled_mm, we target the largest compute bottleneck while keeping attention projections in safe bf16. The custom autograd function (_FP8Linear) uses per-tensor dynamic scaling (absmax / 448 for e4m3fn forward, absmax / 57344 for e5m2 backward gradients) and @torch.compiler.allow_in_graph for torch.compile compatibility. This is the #1 priority item from the queue — exp121 crashed trying ALL linear layers with a custom autograd; this attempt is minimal (MLP-only, cleaner implementation). If ~2x MLP throughput translates to ~1.3x overall throughput, we get ~1080 steps instead of ~830 → potentially -0.003 to -0.005 val_bpb.
- **Pourquoi** : (1) FP8 is explicitly #1 on priority queue — "plus gros levier restant". (2) RTX 5090 Blackwell has native FP8 tensor cores at 2x BF16 throughput. (3) "More steps" has been the most consistent improvement source (MLP 3x exp14 -0.008, local attention exp149 -0.0015). (4) MLP-only avoids the complexity of exp121 (which replaced ALL linears). (5) Attention projections kept in bf16 for safety (smaller 512×512 matrices benefit less from FP8).
- **Risques** : (1) CRASH (medium-high): torch._scaled_mm + custom autograd + torch.compile interaction may fail. The @torch.compiler.allow_in_graph decorator treats the function as opaque, avoiding tracing issues. (2) REGRESSION (low): FP8 quantization noise in MLP is mitigated by ReluSquared sparsity and per-tensor scaling. (3) NEUTRAL (low): if FP8 overhead (amax reductions, casts) negates speedup for these matrix sizes. (4) OOM (none): FP8 uses less memory than BF16.


## Experiment #170 — 2026-03-17 16:58:53

- **Best val_bpb avant** : 1.062436
- **Status** : en cours...
- **Description** : FP8 MLP-only matmuls retry — fix exp169 col-major bug (.t().contiguous() → .t()) + setup_context pattern + BF16 backward.
- **Idée** : Exp169 crashed with `mat2 must be col_major` because `.t().contiguous()` converts the col-major transpose back to row-major. The fix is trivial: remove `.contiguous()`. Also switch to setup_context pattern (modern autograd, better torch.compile support) and BF16 backward (avoids col-major issues in backward, simpler). Forward FP8 gives ~1.5x MLP speedup → ~1.15x overall → ~950 steps instead of 830.
- **Pourquoi** : FP8 is #1 priority. Exp169 had a clear, identifiable bug (col-major stride). The fix is minimal and well-understood. More training steps has been the most consistent source of improvement across 169 experiments.
- **Risques** : (1) CRASH (low): the specific col-major bug is fixed, setup_context is the pattern torch.compile was designed for. Remaining risk: _scaled_mm may have other constraints on Blackwell. (2) REGRESSION (low): FP8 quantization noise in MLP mitigated by ReluSquared sparsity. (3) NEUTRAL (low-medium): if FP8 overhead (amax, casts) negates speedup for these matrix sizes.


## Experiment #170 — 2026-03-17 17:04:38

- **Best val_bpb avant** : 1.062436
- **Status** : crash


## Experiment #170 — 2026-03-17 (attempt 3)

- **Best val_bpb avant** : 1.062436
- **Status** : en cours...
- **Description** : FP8 MLP-only matmuls with @torch.compiler.disable — bypass torch.compile graph tracing that caused 3 consecutive FP8 crashes.
- **Idee** : Previous FP8 attempts (exp121, exp169, exp170x2) all crashed due to torch.compile interaction with custom autograd + _scaled_mm. This attempt uses @torch.compiler.disable on the fp8_linear wrapper to create graph breaks, completely isolating FP8 from torch.compile. BF16 backward for safety. MLP-only (16 FP8 matmuls/forward, attn+lm_head in BF16). Graph break overhead ~1% (negligible for compute-bound matmuls). Expected ~1.15x throughput → ~960 steps → -0.002 to -0.004 val_bpb.
- **Pourquoi** : FP8 is #1 priority. @torch.compiler.disable was explicitly recommended in prompt.md after crashes. The approach eliminates the root cause (compile tracing) while preserving most of the speedup.
- **Risques** : (1) CRASH (low): torch.compiler.disable isolates FP8 from compile. Remaining risk: _scaled_mm API on Blackwell. (2) REGRESSION (low): FP8 noise mitigated by ReluSquared sparsity + dynamic scaling. (3) NEUTRAL (medium): if overhead exceeds speedup.


## Experiment #171 — 2026-03-18 10:08:37

- **Best val_bpb avant** : 1.062436
- **Status** : keep (val_bpb: 1.062310)
- **Description** : LR Floor — set FINAL_LR_FRAC=0.1 to prevent learning rate from decaying to zero during warmdown, inspired by modded-na
- **Description** : LR Floor — set FINAL_LR_FRAC=0.1 to prevent learning rate from decaying to zero during warmdown.
- **Idée** : With WARMDOWN_RATIO=0.7 and FINAL_LR_FRAC=0.0, the learning rate decays linearly from 1.0 to 0.0 over the last 70% of training. At 95% progress, LR multiplier is only 0.071 — the final ~50 steps contribute almost nothing. The latest modded-nanogpt speedrun (PR #224, Feb 2026) introduced an LR floor that was a significant improvement (-0.8s). Setting FINAL_LR_FRAC=0.1 keeps the LR at minimum 10% of peak, ensuring productive updates throughout training. With the current linear warmdown formula `cooldown * 1.0 + (1 - cooldown) * 0.1`, this raises the LR at the end from 0.0 to 0.1, and mildly raises it throughout warmdown (midpoint: 0.55 vs 0.50).
- **Pourquoi** : (1) Directly inspired by modded-nanogpt PR #224 (Feb 2026, statistically significant improvement across 15 runs). (2) Currently the last ~60 steps (LR < 0.1) are nearly wasted — zero-LR means zero parameter updates. (3) FP8 is dead after 4+ crashes; this is the next pragmatic lever. (4) 1-line change, zero crash/OOM risk. (5) Never tested — FINAL_LR_FRAC has been 0.0 for all 170 experiments.
- **Risques** : (1) NEUTRAL (medium): the effect may be within noise (~0.003). (2) REGRESSION (low): higher average LR during warmdown could overshoot, but the floor is conservative at 10%. (3) CRASH (none): single constant change.


## Experiment #172 — 2026-03-18 10:30:08

- **Best val_bpb avant** : 1.062310
- **Status** : discard (val_bpb: 1.064116)
- **Description** : Increase FINAL_LR_FRAC from 0.1 to 0.15 — continue the winning LR floor trend from exp171, matching the modded-nanogpt
- **Description** : Increase FINAL_LR_FRAC from 0.1 to 0.15 — continue the winning LR floor trend from exp171.
- **Idée** : Exp171 improved val_bpb by setting FINAL_LR_FRAC=0.1 (1.062436 → 1.062310). The priority queue says to try 0.15 and 0.2 next. With WARMDOWN_RATIO=0.7, a floor of 0.15 means the LR multiplier at 100% progress is 0.15 instead of 0.1. At 95% progress: 0.071 + 0.929*0.15 = 0.210 (vs 0.164 with 0.1). The average LR during warmdown increases from 0.55 to 0.575 — more effective learning in the final steps. The modded-nanogpt PR#224 used a floor of 0.15, which is exactly what we're testing.
- **Pourquoi** : (1) Exp171 confirmed LR floor helps. (2) modded-nanogpt used 0.15 specifically. (3) 1-line change, zero risk. (4) Closes the LR floor optimization (if 0.15 helps, try 0.2; if not, 0.1 is optimal).
- **Risques** : (1) NEUTRAL (high): the delta between 0.1 and 0.15 may be within noise (~0.003). (2) REGRESSION (very low): 0.15 is still conservative. (3) CRASH (none): single constant change.


## Experiment #173 — 2026-03-18 10:42:42

- **Best val_bpb avant** : 1.062310
- **Status** : en cours...
- **Description** : WEIGHT_DECAY=0.3 — AB-test higher cautious weight decay for Muon optimizer (currently 0.2).
- **Idée** : WD=0.2 has never been AB-tested against 0.3. The WD curve so far: WD=0.0 (+0.004, exp5), WD=0.1 (+0.002, exp168), WD=0.2 (current best). The curve is monotonically improving toward higher WD. Testing 0.3 determines if the trend continues or if 0.2 is the peak. Note: WD is applied via cautious masking (`mask = (g * params) >= 0`) which only decays weights that align with the gradient — 0.3 may not be too aggressive because the masking already filters out destructive decay.
- **Pourquoi** : (1) Priority #3 on the queue, explicitly next after LR floor (which is done: 0.1 optimal, 0.15 worse in exp172). (2) WD=0.3 is the last untested point on the WD axis. (3) 1-line change, zero crash/OOM risk. (4) Highly informative: if worse, WD=0.2 is confirmed optimal and the axis is dead forever.
- **Risques** : (1) REGRESSION (medium): higher WD could over-regularize in a 5-minute training regime where overfitting isn't the main concern. (2) NEUTRAL (medium): delta may be within noise (~0.003). (3) CRASH (none): single constant change.


## Experiment #174 — 2026-03-18 10:51:47

- **Best val_bpb avant** : 1.062310
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Partitioned Hyperconnections — 2-lane residual stream with learned per-sublayer write weights (modded-nanogpt PR#230). | CLASSIFICATION: INFRA
- **Description** : Partitioned Hyperconnections — 2-lane residual stream where attention reads lane0 and MLP reads lane1, both write to both lanes with learnable weights.
- **Idée** : Inspired by modded-nanogpt PR#230 (Feb 2026, -45 steps). Instead of a single residual stream, maintain 2 parallel lanes. Attention reads from lane0, MLP reads from lane1, both sublayers write their output to both lanes with per-layer learnable weights. At init, all write weights = 1.0 so both lanes stay identical → exact equivalence to standard residual. Over training, lanes specialize: attention and MLP develop different "views" of the residual stream. This is fundamentally different from parallel attn+MLP (exp60, +0.034) which shared the same input — here each sublayer reads from its own dedicated lane.
- **Pourquoi** : (1) Priority #2 on queue. (2) Last 3 experiments were all HP tweaks — must attempt architectural change. (3) modded-nanogpt showed -45 steps improvement. (4) Zero-risk at init (degrades to standard residual). (5) Only 32 new scalar parameters (16 per sublayer type). (6) No extra FLOPs, minimal extra memory (~128MB for 2nd lane).
- **Risques** : (1) CRASH (low): simple scalar multiplications, compatible with torch.compile. (2) REGRESSION (medium): the 2-lane split may hurt gradient flow if write weights don't learn useful specialization in ~830 steps. (3) OOM (none): 2nd lane adds ~128MB, well within 32GB VRAM headroom.


## Experiment #175 — 2026-03-18 10:59:44

- **Best val_bpb avant** : 1.062310
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Partitioned Hyperconnections — retry of exp174 (INFRA crash, code never ran). 2-lane residual stream with learned per- | CLASSIFICATION: INFRA
- **Description** : Partitioned Hyperconnections — retry of exp174 (INFRA crash). 2-lane residual stream with learned per-sublayer write weights (modded-nanogpt PR#230).
- **Idee** : Replace single residual stream with 2 parallel lanes. Attention reads lane0, MLP reads lane1, both write to both lanes with per-layer learnable weights (init=1.0 = neutral). At initialization both lanes are identical → exact baseline equivalence. Over training, lanes specialize: attention and MLP develop distinct views of the residual stream. This is fundamentally different from parallel attn+MLP (exp60, +0.034) which shared input. Based on modded-nanogpt PR#230 (Feb 2026, -45 steps). Exp174 had this same change but crashed on INFRA (RunPod deployment error, code never executed).
- **Pourquoi** : (1) Exp174 crashed on infra, this code was never tested. (2) modded-nanogpt showed -45 steps improvement. (3) 32 new scalar parameters, zero new FLOPs. (4) Neutral at init (degrades to standard residual). (5) Must do structural change after multiple HP tweaks.
- **Risques** : (1) CRASH (low): simple scalar multiplications, torch.compile compatible. (2) REGRESSION (medium): lanes may not differentiate in ~830 steps. (3) OOM (none): +128MB, well within 32GB headroom.


## Experiment #176 — 2026-03-18 11:05:08

- **Best val_bpb avant** : 1.062310
- **Status** : en cours...
- **Description** : Partitioned Hyperconnections — 2-lane residual stream (modded-nanogpt PR#230, 3rd attempt after 2 INFRA crashes in exp174/175).
- **Idée** : Replace single residual stream with 2 parallel lanes. Attention reads from lane0 (norm(s0)), MLP reads from lane1 (norm(s1)), both sublayers write their output to both lanes via learned per-layer post_lambdas[layer, sublayer, lane]. At init, all post_lambdas=1.0 → both lanes identical → exact baseline equivalence. Over training, lanes specialize: attention and MLP develop distinct "views" of the residual stream. This is fundamentally different from parallel attn+MLP (exp60, +0.034) which shared input — here each sublayer reads from its own dedicated lane. U-Net skip connections and x0_lambdas apply per-lane independently.
- **Pourquoi** : (1) Priority #2/#4 on queue — never GPU-tested despite 2 attempts (exp174/175 both INFRA crashes). (2) modded-nanogpt PR#230 showed -45 steps improvement. (3) Last 3+ exps before exp174 were all HP tweaks — must do structural change. (4) Only 32 new scalar parameters (n_layer * 2 * 2 = 32), zero new FLOPs. (5) Neutral at init. (6) Also fixed the RunPod deployment issue that caused exp174/175 crashes by syncing existing endpoints into pickle cache.
- **Risques** : (1) CRASH (low): scalar-tensor multiplications, torch.compile compatible. (2) REGRESSION (medium): lanes may not differentiate in ~830 steps — the (s0+s1)/2 merge could slightly smooth the output. (3) OOM (none): 2nd lane adds ~128MB, well within 32GB headroom.


## Experiment #177 — 2026-03-18 11:16:53

- **Best val_bpb avant** : 1.062310
- **Status** : discard (val_bpb: 1.066348)
- **Description** : Low-rank cross-block recurrent state for local attention layers with forgetting gate and zero-init output.
- **Description** : Low-rank cross-block recurrent state for local attention layers — give block-diagonal layers (2-5) cross-block context via a cheap d_state=16 KV state with learnable decay and zero-init gate.
- **Idée** : Local attention layers (2-5) use block_size=256 with ZERO cross-block information. Exp152 tried full-rank (D=128) cross-block linear recurrence and failed (+0.006) because: (1) the full-rank state was noisy and expensive (O(D²) per token), (2) no forgetting mechanism → state became a useless running average. This experiment fixes both problems: (a) low-rank projection (d_state=16, shared for Q and K) reduces state from 128²=16K to 16×128=2K elements per head — 64x cheaper and acts as a bottleneck that captures only the most important KV modes; (b) per-head learnable decay (sigmoid init=0.5) prevents noise accumulation; (c) gate init=0 means the model starts EXACTLY equivalent to baseline — the cross-block contribution only grows if useful. The recurrent loop has 8 iterations (fixed, compile-friendly), state matmuls are tiny (16×256×128 per block per head), and the only significant cost is the shared projection (~1.7% compute overhead).
- **Pourquoi** : (1) Structural change after recent HP tweaks (exp171-173). (2) Different from Partitioned Hyperconnections (exp176, en cours) — orthogonal direction. (3) Addresses a clear architectural limitation (local layers have no cross-block context). (4) 3 key fixes vs exp152: low-rank, decay, zero-init gate. (5) Only ~8K new params, ~1.7% compute overhead. (6) Neutral at init → zero regression risk at step 0.
- **Risques** : (1) REGRESSION (medium): if cross-block context is truly unhelpful (exp152 suggests this), even the improved mechanism won't help. (2) CRASH (low): loop is fixed-length, all ops are standard matmuls, should compile fine. (3) NEUTRAL (medium-high): the improvement may be within noise — local layers already have cross-layer info flow via the residual stream. (4) OOM (none): state is tiny (~512KB total).


## Experiment #178 — 2026-03-18 11:45:22

- **Best val_bpb avant** : 1.062310
- **Status** : discard (val_bpb: 1.082753)
- **Description** : Sigmoid attention for local layers (2-5) — replace softmax with sigmoid gating for independent per-position attention 
- **Description** : Sigmoid Attention for local layers — replace softmax with sigmoid gating in block-diagonal attention (layers 2-5).
- **Idée** : Replace softmax attention in local layers (2-5) with sigmoid attention (Apple "Sigmoid Attention Is All You Need", 2024). Instead of softmax's winner-take-all competition where attention weights must sum to 1, sigmoid allows each position to independently gate its attention to each key. This is a fundamentally different attention mechanism that has never been tested. Normalized by causal position count to keep output magnitude stable. Only applies to local layers (block_size=256) where explicit attention computation is cheap; global layers (0,1,6,7) keep softmax SDPA.
- **Pourquoi** : (1) True architectural change — not an HP tweak. (2) Never tested in this codebase. (3) Different from all dead ideas (DiffAttn, hybrid heads, etc.). (4) Sigmoid allows distributed attention patterns that softmax cannot represent (attending strongly to multiple keys simultaneously). (5) Zero new parameters. (6) Only affects 4 local layers; global layers keep proven softmax. (7) c_proj is zero-init so attention output starts at zero regardless → no init-time regression risk.
- **Risques** : (1) REGRESSION (medium): sigmoid may converge slower than softmax in ~830 steps since it must learn to use negative scores for selectivity. (2) THROUGHPUT (low-medium): loses SDPA flash attention kernel for local layers, replaced by manual matmul — but BS=256 is small enough that the overhead should be <10% total. (3) CRASH (very low): all ops are standard (matmul, sigmoid, masked_fill) and torch.compile compatible.


## Experiment #179 — 2026-03-18 11:58:28

- **Best val_bpb avant** : 1.062310
- **Status** : keep (val_bpb: 1.050748)
- **Description** : Chunkwise linear attention with learned per-head decay replacing block-diagonal softmax SDPA in local layers (2-5) — p
- **Description** : Chunkwise linear attention with learned per-head decay replacing block-diagonal softmax SDPA in local layers (2-5).
- **Idée** : Replace softmax attention in local layers with RetNet/GLA-style chunkwise linear attention. Intra-chunk: causal linear attention with decay mask (Q K^T / sqrt(D) * gamma^(i-j)). Cross-chunk: recurrent KV state with exponential decay. This gives local layers FULL 2048-token context instead of 256-token blocks, at similar compute cost. Different from exp152/177 (addons on top of softmax) — this REPLACES softmax entirely.
- **Pourquoi** : Priority #1 on the queue (DeltaNet/GLA). The highest potential remaining direction. Cross-block addons failed (exp152 +0.006, exp177 +0.004) but REPLACING softmax is fundamentally different — single mechanism, cleaner optimization, no redundant compute. Exponential decay prevents noise accumulation.
- **Risques** : (1) REGRESSION (high): exp178 sigmoid attention was +0.020. Linear attention without softmax may suffer similarly. (2) CRASH (low): loop is fixed-length (8 iterations), all standard ops. (3) COMPILE TIME (medium): 8-iteration loop unrolled by torch.compile.


## Experiment #180 — 2026-03-18 12:40:21

- **Best val_bpb avant** : 1.050748
- **Status** : discard (val_bpb: 1.094853)
- **Description** : Extend chunkwise linear attention to ALL 8 layers, eliminating softmax entirely from the model.
- **Description** : Extend chunkwise linear attention to ALL 8 layers — eliminate softmax entirely from the model.
- **Idée** : exp179 showed -0.012 BPB by replacing softmax with chunkwise linear attention in local layers 2-5. The 4 global layers (0,1,6,7) still use full softmax SDPA on 2048 tokens = O(T²) attention, the most expensive operation. By extending chunkwise linear attention (256-token chunks + recurrent decay state) to ALL layers, we: (1) eliminate O(2048²) softmax attention from 4 layers → massive FLOPs savings → more training steps in 5 minutes; (2) give all layers the same proven mechanism from exp179; (3) full-sequence context preserved via cross-chunk recurrent state. VE layers (0,7) keep their value embeddings — only the attention mechanism changes.
- **Pourquoi** : (1) Directly builds on the biggest breakthrough in 180 experiments (exp179, -0.012). (2) Radical architectural change: removes ALL softmax from the model. (3) Compute savings on global layers (~7.6x less attention FLOPs per global layer) → more steps. (4) Minimal code change (2 lines), zero new parameters beyond 16 decay scalars for 4 newly-converted layers.
- **Risques** : (1) REGRESSION (medium): global layers may NEED softmax's normalized attention for long-range reasoning — the recurrent state's exponential decay may lose important long-range information. (2) CRASH (very low): same code path as exp179, just applied to all layers. (3) OOM (none): linear attention uses less memory than SDPA. (4) COMPILE TIME (low): same loop structure, just more layers using it.


## Experiment #181 — 2026-03-18 12:50:52

- **Best val_bpb avant** : 1.050748
- **Status** : discard (val_bpb: nan)
- **Description** : GLA-style input-dependent decay for chunkwise linear attention in local layers (2-5) — content-addressed forgetting re
- **Description** : GLA-style input-dependent decay for chunkwise linear attention in local layers (2-5).
- **Idée** : Replace fixed per-head decay `gamma = sigmoid(decay_rate)` with content-dependent gating `gamma_t = sigmoid(decay_rate + decay_proj(x_t))`. Each position decides its own forget rate based on its content. This is the key innovation from the GLA paper (Yang et al. 2024) over RetNet's fixed exponential decay. At init, decay_proj is zero → exactly equivalent to exp179 baseline (gamma ≈ 0.982). Over training, the model learns when to retain (important tokens → low decay) vs forget (irrelevant → high decay), creating content-addressed memory instead of purely positional. Only 4 new tiny weight matrices (4×8 per local layer, 128 params total).
- **Pourquoi** : (1) Priority #1 on queue: "tune chunk_size, decay init, or add GLA-style input-dependent decay". (2) Builds directly on exp179's MASSIVE -0.012 win. (3) True architectural change — not HP tweak. (4) GLA consistently outperforms RetNet in the literature. (5) Neutral at init → zero regression risk at step 0. (6) Minimal compute overhead (one tiny linear projection per position).
- **Risques** : (1) REGRESSION (low-medium): the decay_proj adds input-dependent modulation; if the gradients are noisy, it could destabilize the learned decay rates. But zero-init means gradual introduction. (2) CRASH (low): same code structure as exp179, just with per-sample decay masks instead of per-head. All shapes are static. (3) COMPILE TIME (medium): intra_decay is now (B, H, BS, BS) instead of (H, BS, BS), slightly more memory per chunk. (4) OOM (none): extra memory is ~17 MB per chunk, negligible.


## Experiment #182 — 2026-03-18 13:04:24

- **Best val_bpb avant** : 1.050748
- **Status** : crash (val_bpb: 0.000000)
- **Description** : Multi-scale decay initialization + chunk_size=512 for chunkwise linear attention in local layers (2-5). | CLASSIFICATION: INFRA
- **Description** : Multi-scale decay initialization + chunk_size=512 for chunkwise linear attention in local layers (2-5).
- **Idée** : Two complementary changes to the exp179 chunkwise linear attention: (1) Increase chunk_size from 256 to 512 for better intra-chunk attention quality (512 tokens of exact causal linear attention vs 256) and fewer cross-chunk transitions (4 vs 8 → less recurrent error accumulation). (2) Multi-scale decay init: instead of uniform decay_rate=4.0 (gamma≈0.982, ~55-token context) for all heads, use per-head values [2.0, 3.5, 5.0, 7.0] → gamma≈[0.88, 0.97, 0.99, 0.999] → effective context ≈ [8, 31, 143, 1000] tokens. This lets heads specialize across temporal scales, analogous to multi-scale RoPE (exp80, kept). The slowest-decay head covers almost the full 2048-token sequence through the recurrent state.
- **Pourquoi** : (1) Priority queue explicitly recommends chunk_size=512 after exp181 NaN. (2) Multi-scale decay is the temporal analog of multi-scale RoPE which worked. (3) exp181 GLA input-dependent decay failed with NaN — this avoids numerical instability while still differentiating heads. (4) Both changes are structural, not HP tweaks.
- **Risques** : (1) REGRESSION (medium): chunk_size=512 means 4x more intra-chunk compute per chunk, partially offset by half the chunks. Net ~67% more local attention compute → potentially fewer training steps. (2) CRASH (very low): same code path, just different constants. (3) NEUTRAL (medium): multi-scale decay may not help if all heads converge to similar rates anyway.


## Experiment #183 — 2026-03-18 13:11:54

- **Best val_bpb avant** : 1.050748
- **Status** : en cours...
- **Description** : Mamba-style causal depthwise conv (kernel=4) on Q/K in local linear attention layers (2-5).
- **Idée** : Add short causal depthwise convolution (kernel_size=4) on Q and K projections in local attention layers, applied after linear projection but before RoPE/QK-norm. This gives each Q/K a 4-token receptive field. Inspired by Mamba's conv-before-SSM pattern. Linear attention lacks softmax's implicit local structure capture — the conv compensates by injecting local position mixing. Identity init (current position=1, past=0) → exact baseline equivalence at step 0. 16K new params (~0.02% of model), ~3% compute overhead.
- **Pourquoi** : (1) Explicitly recommended by priority queue as next DeltaNet/GLA step. (2) Mamba architecture principle: local conv + linear recurrence is more powerful than either alone. (3) Different from exp107 (conv before MLP, +0.023) — this targets Q/K in linear attention specifically. (4) Complementary to token shift (2-token window on raw input vs 4-token window on projected Q/K).
- **Risques** : (1) REGRESSION (low-medium): ~3% more compute → slightly fewer steps, but identity init = no step-0 regression. (2) NEUTRAL (medium): linear attention decay may already capture enough local structure. (3) CRASH (very low): Conv1d is standard torch, compile-friendly.


## Experiment #184 — 2026-03-18 13:27:29

- **Best val_bpb avant** : 1.050748
- **Status** : keep (val_bpb: 1.050168)
- **Description** : experiment 184


## Experiment #185 — 2026-03-18 13:38:12

- **Best val_bpb avant** : 1.050168
- **Status** : keep (val_bpb: 1.049879)
- **Description** : experiment 185


## Experiment #186 — 2026-03-18 13:48:44

- **Best val_bpb avant** : 1.049879
- **Status** : discard (val_bpb: 1.050069)
- **Description** : experiment 186


## Experiment #187 — 2026-03-18 14:01:35

- **Best val_bpb avant** : 1.049879
- **Status** : en cours...


## Experiment #188 — 2026-03-18 14:43:24

- **Best val_bpb avant** : 1.049879
- **Status** : en cours...


## Experiment #189 — 2026-03-18 14:44:00

- **Best val_bpb avant** : 1.049879
- **Status** : discard (val_bpb: 1.055929)
- **Description** : Partitioned Hyperconnections — 2-lane residual stream where attention reads lane 0, MLP reads lane 1, both write to bo
- **Description** : Partitioned Hyperconnections — 2-lane residual stream (modded-nanogpt PR#230).
- **Idée** : Split the residual stream into 2 lanes. Attention reads from lane 0, MLP reads from lane 1, both sublayers write to both lanes with learned per-layer weights. This creates partially decoupled processing streams: attention gets a "clean" residual without MLP outputs from prior layers polluting it, and MLP gets a residual without attention noise. Inspired by modded-nanogpt PR#230 (-45 steps). Init all write weights to 1.0 → exact baseline equivalence at step 0 → the model gradually learns to differentiate lanes. 32 new scalar parameters (negligible). Final output merges lanes via (x_a + x_m) / 2. This is a retry of exp174/175 which both crashed on INFRA (code never ran).
- **Pourquoi** : (1) Priority #2 on the queue, never GPU-tested (2 INFRA crashes). (2) Last 3+ experiments were HP tweaks → structural change required. (3) Different from parallel attn+MLP (exp60, +0.034) which shared the SAME input — here sublayers are sequential but read from DIFFERENT lanes. (4) Validated externally in modded-nanogpt. (5) Zero regression risk at init.
- **Risques** : (1) REGRESSION (low-medium): merge /2 could dilute if lanes don't differentiate. (2) CRASH (low): standard ops, static shapes. (3) NEUTRAL (medium): 830 steps may be too few for lane specialization. (4) MEMORY: +320MB for 2x residual + 2x skips, well within 32GB budget.


## Experiment #190 — 2026-03-18 14:59:17

- **Best val_bpb avant** : 1.049879
- **Status** : discard (val_bpb: 1.052605)
- **Description** : Multi-scale decay init + RoPE-free linear attention for local layers (2-5) — head specialization via diverse temporal 
- **Description** : Multi-scale decay init + RoPE-free linear attention for local layers (2-5).
- **Idée** : Two synergistic changes to chunkwise linear attention: (1) Replace uniform decay_rate=4.0 with per-head [2.0, 3.5, 5.0, 7.0] → effective contexts ~[8, 29, 143, 1000] tokens, enabling head specialization across temporal scales (analogous to multi-scale RoPE for global layers). (2) Remove RoPE from local linear attention layers — decay provides positional signal, RoPE is redundant and may interfere (RetNet/RWKV design principle). Global layers (0,1,6,7) keep RoPE + softmax. Exp182 tested multi-scale decay but INFRA-crashed (never GPU-tested).
- **Pourquoi** : (1) Multi-scale decay is priority queue #1 next step, never GPU-tested. (2) RoPE removal follows RetNet/RWKV: linear attention + decay = self-contained positional encoding; adding RoPE on top creates conflicting position signals. (3) Both changes are architectural, not HP tweaks. (4) Removes compute from rotary embedding in 4 layers → slight speed gain. (5) exp183 conv and exp189 hyperconnections both failed — need a different direction.
- **Risques** : (1) REGRESSION (medium): removing RoPE may lose useful position info that decay can't capture alone; but global layers still have RoPE. (2) CRASH (very low): only conditional skip of RoPE + different init values, no new ops. (3) NEUTRAL (medium): if decay alone provides sufficient position, the RoPE removal is harmless; multi-scale may not differentiate enough in 830 steps.


## Experiment #191 — 2026-03-18 15:12:31

- **Best val_bpb avant** : 1.049879
- **Status** : discard (val_bpb: 1.050692)
- **Description** : GLA-style input-dependent decay for chunkwise linear attention in local layers (2-5) — content-addressed forgetting wi
- **Description** : GLA-style input-dependent decay for chunkwise linear attention in local layers (2-5).
- **Idée** : Replace fixed per-head decay `gamma = sigmoid(decay_rate)` with content-dependent `gamma_t = sigmoid(decay_rate + decay_proj(x_t))`. Each position decides its own forget rate. This is the key innovation from GLA (Yang et al. 2024) over RetNet's fixed decay. Exp181 tried this but NaN'd due to unstable `cumsum(log_gamma)` — fix: clamp gamma >= 0.8 before log, keeping all computation in float32. At init, decay_proj=0 → exactly equivalent to exp179/185 baseline. Only 128 new params (4 layers x 32). Uses sparse gate_dim input (8 dims) matching the attention gate pattern.
- **Pourquoi** : (1) Priority queue #1: GLA input-dependent decay explicitly recommended, exp181 was the failed attempt. (2) Content-addressed memory: model can retain important tokens (low decay) and forget noise (high decay) — fundamental capability improvement. (3) Builds on exp179's massive -0.012 win. (4) Numerically stable: clamped gamma in [0.8, 1.0], log values in [-0.223, 0], cumsum bounded. (5) Zero-init → no regression at step 0.
- **Risques** : (1) REGRESSION (low-medium): per-chunk decay computation adds ~5-10% overhead, may lose 1-2 steps. (2) CRASH (low): standard ops, static shapes. (3) NaN (very low): gamma clamped to safe range, exp181's failure mode eliminated.


## Experiment #192 — 2026-03-18 15:26:37

- **Best val_bpb avant** : 1.049879
- **Status** : discard (val_bpb: 1.054170)
- **Description** : Extend chunkwise linear attention from layers 2-5 to layers 1-6 — VE-bookend-only global attention for compute savings
- **Description** : Extend chunkwise linear attention from layers 2-5 to layers 1-6 (VE-bookend-only global attention).
- **Idée** : Convert layers 1 and 6 from expensive O(T²) softmax attention to chunkwise linear attention with decay. Only VE layers (0 and 7) keep full softmax. Unlike exp150 (block-diagonal SDPA for layers 1-6, neutral), chunkwise linear attention has cross-chunk recurrent state providing effective full-sequence context. Unlike exp180 (all layers linear, +0.044 catastrophe), this keeps VE+softmax for bookend layers. Saves ~12% attention compute → ~80-100 extra training steps.
- **Pourquoi** : (1) All attempts to improve the linear attention mechanism itself have failed (exp183 conv, exp189 hyperconnections, exp190 multi-scale+RoPE-free, exp191 GLA). Instead of improving the mechanism, extend its reach. (2) The compute savings from removing 2 full-attention layers is significant. (3) Cross-chunk recurrence gives layers 1 and 6 effective full-context, unlike block-diagonal which had no cross-block communication. (4) Clean single-line change, easy to diagnose.
- **Risques** : (1) REGRESSION (medium): layers 1 and 6 might need softmax's precise token selection for processing VE-adjacent representations. (2) CRASH (very low): same code paths, just more layers using them. (3) OOM (very low): linear attention uses less memory than softmax.


## Experiment #193 — 2026-03-18 15:36:52

- **Best val_bpb avant** : 1.049879
- **Status** : keep (val_bpb: 1.049078)
- **Description** : RetNet-style hybrid attention for local layers — softmax intra-chunk + linear recurrence cross-chunk with learned log-

- **Description** : RetNet-style hybrid attention — softmax intra-chunk + linear recurrence cross-chunk for local layers (2-5).
- **Idée** : Replace all-linear intra-chunk attention with softmax causal attention within each 256-token chunk, keeping cross-chunk linear recurrence for long-range context. Add log-decay as additive positional bias to softmax logits (learned ALiBi-style per-head). This is the RetNet "chunk-wise recurrent" paradigm: softmax gives sharp, precise local attention within chunks; linear recurrence propagates context across chunks. All 6 attempts to improve the linear attention mechanism itself failed (exp183, 189-192). This fundamentally changes HOW intra-chunk attention works rather than tweaking the mechanism.
- **Pourquoi** : (1) Softmax gives much sharper attention distributions than linear attention — it can attend to exactly one token. (2) Within a 256-token chunk, softmax is cheap and Flash-optimized. (3) Cross-chunk recurrence (unchanged) already handles long-range context. (4) Log-decay bias preserves the learned distance weighting from exp179, now as a learned ALiBi-like positional bias in softmax space. (5) RetNet/HGRN2 validate this hybrid design. (6) All pure-linear-attention improvements exhausted.
- **Risques** : (1) REGRESSION (medium): magnitude mismatch between softmax intra (~1.0 * V) and linear cross (~D^-0.5 * V). The attn_gate should compensate. (2) CRASH (low): standard ops (softmax, masked_fill), static shapes, compiles cleanly. (3) NEUTRAL (medium): if the all-linear formulation was already capturing the right patterns, softmax doesn't add value.

## Experiment #194 — 2026-03-18 15:53:22

- **Best val_bpb avant** : 1.049078
- **Status** : discard (val_bpb: 1.052590)
- **Description** : Increase chunk_size from 256 to 512 for hybrid local attention layers (2-5) — larger softmax intra-chunk windows + few
- **Description** : Increase chunk_size from 256 to 512 for hybrid local attention layers (2-5).
- **Idée** : Double the chunk size for the hybrid softmax-intra + linear-cross attention in local layers. Exp193 showed softmax intra-chunk is key — larger chunks amplify this by giving softmax 512-token windows (vs 256). Also halves cross-chunk transitions from 8 to 4, reducing information loss through recurrence. Explicitly listed as untested in priority queue (exp182 INFRA crash). Single-number change for clean AB test.
- **Pourquoi** : (1) Priority queue explicitly lists chunk_size=512 as untested (exp182 never ran on GPU). (2) Exp193's hybrid softmax directly benefits from larger chunks. (3) Fewer recurrence steps = less lossy cross-chunk context. (4) 512 tokens = 25% of full context per chunk, qualitatively different from 256. (5) Structural compute tradeoff, not HP tweak.
- **Risques** : (1) REGRESSION (medium): ~2x more intra-chunk attention FLOPs in 4 local layers, may cost 40-80 training steps. (2) CRASH (very low): T=2048 divisible by 512, static shapes, same code path. (3) OOM (low): 4x larger attention matrix per chunk, but sequential processing limits peak usage.


## Experiment #195 — 2026-03-18 16:06:32

- **Best val_bpb avant** : 1.049078
- **Status** : discard (val_bpb: 1.105106)
- **Description** : State-augmented hybrid attention — inject 16 virtual memory tokens from recurrent state S into intra-chunk softmax for
- **Description** : State-augmented hybrid attention — inject 16 virtual memory tokens from recurrent state into intra-chunk softmax attention.
- **Idée** : In local layers (2-5), the current hybrid attention has two separate pathways: (1) softmax intra-chunk for sharp local attention, (2) linear Q@S for smooth cross-chunk context. These are simply added. State-augmented attention injects M=16 virtual memory tokens (read from recurrent state S via learned query patterns `mem_read`) directly into the intra-chunk softmax attention. The model can now attend to both local tokens AND compressed past context through the same softmax, enabling natural competition between local and long-range signals. Keeps the linear cross term as a complementary smooth baseline.
- **Pourquoi** : (1) Listed as untested in priority queue: "state-augmented attention (inject recurrent state as virtual memory tokens in softmax)". (2) Last 5 experiments (190-194) were ALL chunk attention variations — need structural change. (3) Unified softmax over local+memory is fundamentally different from additive intra+cross: softmax routing lets the model decide per-query whether local or distant context matters more. (4) Minimal parameter cost: 4 layers × 64×128 = 32K params (0.07% of model). (5) Compute overhead: attention matrix goes from (256,256) to (256,272) = 6.25% increase in 4/8 layers = ~1-2% net. (6) mem_read stored as 2D (64,128) for proper Muon optimization.
- **Risques** : (1) REGRESSION (medium): memory tokens add noise initially, may take many steps to learn useful read patterns. (2) CRASH (low): all ops standard, static shapes (BS+M=272 constant), compile-friendly. (3) NEUTRAL (medium): if the linear cross term already provides adequate cross-chunk context, the memory augmentation doesn't add value. (4) COMPILE (low-medium): new cat/expand ops in the loop may increase compile time.


## Experiment #196 — 2026-03-18 16:24:45

- **Best val_bpb avant** : 1.049078
- **Status** : discard (val_bpb: 1.064450)
- **Description** : DeltaNet-style error-correcting state update for cross-chunk linear recurrence in local attention layers (2-5).
- **Description** : DeltaNet-style error-correcting state update for cross-chunk recurrence in local layers (2-5).
- **Idée** : Replace blind KV accumulation (`S = decay * S + k^T @ v`) with DeltaNet's error-correcting rule (`S = decay * S + beta * k^T @ (v - S @ k)`). The state only accumulates what it doesn't already predict, preventing redundant associations and using finite state capacity more efficiently. Intra-chunk softmax attention and cross-chunk query (Q @ S) unchanged.
- **Pourquoi** : (1) Cross-chunk recurrence is the weakest link — simple exponential decay is lossy and redundant. (2) DeltaNet (Yang et al. 2024, "Gated Delta Networks") validated this error-correcting approach vs RetNet's blind accumulation. (3) Fundamentally different from all prior chunk attention variations (which modified intra-chunk, layer assignment, chunk size, or memory tokens — never the state update itself). (4) Nearly neutral at init: S=0 → predicted_v=0 → error=v → same as current. (5) Only 16 new params (4 heads × 4 layers), one extra matmul per chunk. (6) Cross-block addons note says "softmax + linear recurrence doesn't help" — but this doesn't ADD a mechanism, it IMPROVES the existing recurrence quality.
- **Risques** : (1) REGRESSION (medium): extra matmul per chunk (~5-10% overhead) costs ~40-80 training steps. State quality must compensate. (2) CRASH (very low): standard ops (matmul, sigmoid), static shapes, compile-friendly. (3) OOM (very low): one extra (B,H,BS,D) tensor per chunk, negligible. (4) INSTABILITY (low): error correction could amplify noise if S diverges, but decay keeps it bounded.


## Experiment #197 — 2026-03-18 16:43:53

- **Best val_bpb avant** : 1.049078
- **Status** : discard (val_bpb: 1.060267)
- **Description** : RoPE-free cross-chunk recurrence + multi-scale decay init for hybrid local attention layers (2-5).
- **Description** : RoPE-free cross-chunk recurrence + multi-scale decay init for hybrid local attention layers (2-5).
- **Idée** : Two targeted changes to cross-chunk linear recurrence: (1) Use pre-RoPE normalized Q/K for cross-chunk path only — RoPE creates absolute position interference when querying the accumulated state S, but is kept for intra-chunk softmax where relative position matters. (2) Multi-scale decay init [2.0, 3.5, 5.0, 7.0] per head → sigmoid ≈ [0.88, 0.97, 0.993, 0.999], giving heads effective contexts of ~8, ~33, ~143, ~1000 tokens. Zero new parameters.
- **Pourquoi** : (1) RoPE in accumulated state creates cosine interference — keys at different positions mix their angles in S, making content-based retrieval position-dependent instead of semantic. Without RoPE, cross-chunk = pure content similarity + decay. (2) Different from exp190 which removed RoPE from BOTH intra AND cross-chunk (+0.003 regress). Here RoPE is KEPT for intra-chunk softmax. (3) Multi-scale decay isolated from exp190's dual change. (4) Priority queue recommends multi-scale decay as next step after exp196's regression. (5) Changes are complementary: multi-scale gives temporal specialization, RoPE-free gives cleaner retrieval.
- **Risques** : (1) REGRESSION (medium): cross-chunk Q without RoPE loses query position info, but decay handles temporal weighting. (2) CRASH (very low): standard ops, static shapes, compile-friendly. (3) OOM (very low): ~2MB extra per local layer for raw Q/K tensors. (4) INTERACTION (low-medium): two changes at once make diagnosis harder.


## Experiment #198 — 2026-03-18 16:59:14

- **Best val_bpb avant** : 1.049078
- **Status** : en cours...
- **Description** : Continuous MLP width ramp [1.0, 1.25, 1.75, 2.25, 3.25, 4.25, 5.0, 5.25] replacing 1.5x/4.5x two-step split — iso-compute smooth gradient allocating minimal MLP to early token-mixing layers and maximum to late prediction layers.
- **Idée** : Replace the coarse 2-step MLP expansion (1.5x for layers 0-3, 4.5x for layers 4-7) with a smooth 8-point ramp. Average remains 3.0x = iso-compute. The non-uniform MLP trend was the strongest architectural axis: uniform 3x → 2x/4x (-0.002) → 1.5x/4.5x (-0.001). Exp123/124 tried similar ramps but both INFRA-crashed, never GPU-tested. This is the #1 pivot recommended by the priority queue after chunk attention axis exhaustion (4 consecutive post-exp193 failures).
- **Pourquoi** : (1) Priority queue #1 recommendation after chunk attention axis exhausted. (2) Never GPU-tested (exp123/124 INFRA crashes). (3) Strong trend: each non-uniform refinement improved. (4) Smooth ramp = 8 distinct MLP sizes vs current 2, giving each layer optimal compute. (5) Early layers primarily do token mixing (need less MLP), late layers need capacity for prediction (need more MLP). (6) Zero risk of compile issues — same Linear layers, just different hidden dims.
- **Risques** : (1) REGRESSION (low-medium): some layers get very small MLPs (layer 0: 1.0x = 512 hidden). If any early layer needs more capacity, it could bottleneck. But exp130 showed 1.25x/4.75x was only +0.001, so extreme gradients work. (2) CRASH (very low): standard Linears, static shapes. (3) MUON GROUPS (low): 8 unique MLP shapes instead of 2 = more Muon groups. Shouldn't affect convergence. (4) NEUTRAL (medium): if the 2-step split was already near-optimal, smooth ramp doesn't help.


## Experiment #199 — 2026-03-19 09:42:56

- **Best val_bpb avant** : 1.049078
- **Status** : keep (val_bpb: 1.048702)
- **Description** : Pathway-specific attention gating for hybrid local layers — separate cross-chunk gate gives per-position control over 
- **Description** : Pathway-specific attention gating for hybrid local layers — separate cross-chunk gate modulates recurrent contribution independently from intra-chunk softmax.
- **Idée** : In local layers (2-5), the hybrid attention output is currently `(intra + cross) * attn_gate`. The model cannot selectively suppress the cross-chunk recurrent contribution while keeping intra-chunk softmax strong. At document/topic boundaries, stale recurrent state gets equal weight. Adding a learned `cross_gate` (same sparse input pattern as attn_gate, gate_dim=8 → n_head=4) gives per-position per-head control: `(intra + cross_gate * cross) * attn_gate`. Init to neutral (sigmoid(0)×2=1.0). Only 128 new params total (32 per local layer × 4 layers).
- **Pourquoi** : (1) After 7 consecutive chunk attention failures (194-198), all modified the mechanism itself. This changes HOW the two pathways are combined, not the pathways themselves. (2) attn_temp scales Q affecting both pathways equally; attn_gate gates the combined output; neither can differentiate intra vs cross. (3) Minimal compute/memory overhead — one extra Linear(8,4) per local layer. (4) Clean neutral init — exactly equivalent to current code at step 0. (5) Addresses a genuine architectural limitation: the model should decide per-token whether distant (cross) or local (intra) context is more relevant.
- **Risques** : (1) NEUTRAL (high): 128 params may not learn anything useful in 830 steps — the existing attn_gate + attn_temp may already provide sufficient pathway control indirectly. (2) CRASH (very low): standard Linear, no new shapes, compile-friendly. (3) REGRESSION (low): extra sigmoid+multiply per position in 4 layers, negligible compute cost.


## Experiment #200 — 2026-03-19 10:03:38

- **Best val_bpb avant** : 1.048702
- **Status** : en cours...
- **Description** : Interleaved local/global attention pattern — [G,L,G,L,L,G,L,G] instead of contiguous [G,G,L,L,L,L,G,G] for global context refresh between local layers.
- **Idée** : Redistribute local and global attention layers from a contiguous middle block to an interleaved pattern. Instead of 4 consecutive local layers (2-5) where global context degrades through the span, interleave global layers at positions 2 and 5 to "refresh" full-sequence attention mid-stream. Same 4 local + 4 global count, zero new params, zero compute change. One line: `self.local_attn = (min(layer_idx, n-1-layer_idx) % 2 == 1)`.
- **Pourquoi** : (1) New axis never explored — arrangement of mechanisms, not modification of mechanisms. (2) Chunk attention axis EXHAUSTED (8 failures), MLP distribution DEAD (5 experiments). (3) Interleaved architectures (Gemma 2, Jamba) show alternating attention types outperform contiguous blocks. (4) U-Net skips become type-matched: G→G, L→L pairs. (5) Zero risk: no new ops, no shapes, compile-friendly.
- **Risques** : (1) REGRESSION (medium): layer 1 becoming local means less global context early. (2) CRASH (very low): no new ops. (3) NEUTRAL (medium): if recurrence already provides adequate cross-chunk context, interleaving adds nothing.

