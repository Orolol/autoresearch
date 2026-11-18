DEPTH=9 + GQA (n_kv_head=2) + MLP 3x — iso-compute reallocation from attention to MLP, preserving the winning MLP 3x while gaining depth.

## What changed
- DEPTH: 8 -> 9 (one extra transformer layer)
- ASPECT_RATIO: 64 -> 56 (keeps model_dim at 512 after rounding: 9*56=504 -> 512)
- N_KV_HEAD: 4 -> 2 (GQA: 2 KV heads shared by 4 query heads)
- MLP: stays at 3x (1536 hidden) — the winning configuration from exp14

## Why
- **Exp22 (DEPTH=9 + MLP 2.5x) was tantalizingly close** at 1.080504 (+0.0009 from best). It proved that 9 layers almost compensate for narrower MLP at iso-compute. The key insight: it used the WRONG tradeoff — trading MLP width for depth, when MLP 3x was the breakthrough.
- **This experiment uses the RIGHT tradeoff**: trade attention capacity (via GQA) for depth, while preserving the winning MLP 3x. GQA was essentially free at 8 layers (exp4: +0.0003). Attention KV diversity is cheap at this scale.
- **Exactly iso-compute with exp22**: both have 2,359,296 matrix params per layer (21,233,664 total for 9 layers). But the allocation differs:
  - Exp22: 1,048,576 attention + 1,310,720 MLP (2.5x)
  - This: 786,432 attention + 1,572,864 MLP (3x)
- **Pure architecture comparison**: same total FLOPs, same number of layers, same model dim. The only difference is where compute is allocated — attention vs MLP.

## VE impact analysis
- GQA reduces VE dim from 512 to 256 (n_kv_head * head_dim = 2 * 128)
- VE at 256 was tested in exp4 (GQA at 8 layers) and quality was preserved (+0.0003)
- VE is on the last layer only — 1 layer out of 9 affected
- ve_gate becomes nn.Linear(32, 2) instead of nn.Linear(32, 4) — trivial

## Expected outcome
- If MLP-heavy allocation > attention-heavy allocation: improvement of ~0.001-0.003 over exp22 (beating 1.080504, potentially beating 1.079602)
- If equivalent: neutral within noise (~0.003)
- If GQA hurts at 9 layers or VE at 256 is insufficient: mild regression (~0.001-0.003)
- No crash, OOM, or speed risk — pure iso-compute architecture swap


## Résultats
- val_bpb: 1.085796
- peak_vram_mb: 11096.9
- status: discarded
