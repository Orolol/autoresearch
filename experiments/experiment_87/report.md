Increase DEPTH from 8 to 9 layers (ASPECT_RATIO=56) — retry of exp86 which crashed on infra, never tested on GPU.

## What changed

Two constant changes in train.py:
- `DEPTH = 9` (was 8)
- `ASPECT_RATIO = 56` (was 64, to keep model_dim=512 after rounding)

Result: 9 transformer layers instead of 8, all at 512-dim with 4 heads. Everything adapts automatically:
- U-Net skips: 4 early layers (0-3) save, 4 late layers (5-8) receive, layer 4 is middle
- VE on layers 0 and 8 (shared weights, separate gates)
- Multi-scale RoPE: 4 bases for 4 heads, unchanged
- Window pattern SSSL cycles over 9 layers: S,S,S,L,S,S,S,L,L (last forced to L)

## Why

Depth is the single most valuable architectural axis:
- exp18 (DEPTH=7): +0.033 catastrophe
- exp22 (DEPTH=9 before improvements): 1.080504 (+0.001 from that era's baseline)
- exp86 (DEPTH=9 with improvements): crashed on infra, never tested on GPU

Since exp22, accumulated gradient flow improvements total ~-0.008:
1. Multi-scale RoPE (exp80: -0.002) — head specialization helps deeper models
2. U-Net skips (exp71: -0.001) — now span 9 layers with 4 skip connections
3. VE bookend (exp84: -0.004) — token identity at both extremes of deeper network
4. Softcap 13 (exp45: -0.001) — better gradient flow through logits

Also exp22 used MLP 2.5x (known -0.003 worse than 3x from exp15). This experiment uses MLP 3x.

Back-of-envelope: 1.080504 - 0.008 - 0.003 ≈ 1.070, which would be a new best.

The compute overhead is ~8-10% (embeddings and lm_head are fixed costs), meaning ~750 steps instead of ~830.

## Risks

- Zero crash/OOM risk (standard Block addition, no new code)
- ~8-10% fewer training steps (~750 vs ~830) due to extra layer compute
- If step reduction outweighs capacity gain: mild regression ~0.001-0.003
- Exp86 crashed on RunPod infra (not GPU code issue), this is a retry

## If it works

- Try DEPTH=10 (push depth further with gradient flow improvements)
- Try reducing MLP from 3x to 2.75x to recover some steps with 9 layers
- Try wider RoPE base range (1000, 10000, 100000, 1000000) for more aggressive head specialization across deeper network
- Try adding VE to middle layer (layer 4, U-Net inflection point) in addition to bookend

## If it fails

- 8 layers is the optimal depth for 300s budget even with improved gradient flow
- Try skip_lambdas init 0.05 (exp82 crashed, never tested on GPU)
- Try wider multi-scale RoPE range (1000, 5000, 50000, 500000)
- Try DeltaNet or GLA (replace softmax attention entirely)
- Try per-sublayer output scaling (attn_alpha, mlp_alpha per layer)
