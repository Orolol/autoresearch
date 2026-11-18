Schedule-Free AdamW (Defazio & Mishchenko 2024) for embedding/scalar optimizer groups, replacing linear warmdown with Polyak-Ruppert iterate averaging.

## What changed

Five changes in `train.py`:

1. **Replaced `adamw_step_fused` with `schedule_free_adamw_step`** (compiled function): Instead of standard Adam update on `p` directly, the new function maintains two sequences:
   - `z`: the Adam gradient trajectory (receives standard Adam updates)
   - `x`: Polyak-Ruppert running average (`x.lerp_(z, 1/step)`)
   - `p`: set to interpolated point `y = (1-β₁)z + β₁x` for training

2. **Modified `_step_adamw`**: Initializes `z` and `x` states (cloned from initial parameters) on first step. Calls `schedule_free_adamw_step` instead of `adamw_step_fused`.

3. **Added `eval_mode()` / `train_mode()` to `MuonAdamW`**: Before evaluation, switches all AdamW parameters from training interpolation `y` to the Polyak-averaged `x`. This is the key insight of Schedule-Free: the averaged iterate `x` is the optimal evaluation point.

4. **LR schedule**: AdamW groups now use **constant LR** (no warmdown). The Polyak averaging provides an implicit optimal schedule. Muon groups keep their existing linear warmdown unchanged.

5. **Before final eval**: `optimizer.eval_mode()` is called to switch to averaged parameters.

## Why

- **60% warmdown underutilizes AdamW groups**: With WARMDOWN_RATIO=0.6, the embedding and lm_head LRs decay linearly for 60% of training. Schedule-Free maintains full LR throughout — the averaging handles the "cooling" implicitly.
- **Theoretical optimality**: Schedule-Free Adam provably matches the best possible LR schedule for convex objectives (Defazio & Mishchenko 2024). Empirically matches or beats cosine/linear schedules.
- **Different from exp70** (decouple warmdown, +0.013): Exp70 removed warmdown for AdamW without any compensating mechanism. Schedule-Free replaces warmdown with iterate averaging, which provides the stability that raw constant-LR lacks.
- **Muon untouched**: Muon's optimization dynamics are well-tuned (prompt says "Muon est parfaitement tuné"). Only AdamW groups are modified.
- **Modest memory overhead**: ~400MB for z,x clones of embedding/lm_head params. Negligible on 32GB.

## Risks

- ~830 steps may be too few for Polyak averaging (1/t weight: 1/830 ≈ 0.0012 at final step)
- Constant LR could destabilize embeddings if averaging doesn't compensate
- The y interpolation point during training might produce worse gradients than scheduled parameters
- Extra torch.compile graph for the new function signature
- Expected range: -0.008 to +0.005

## If it works

- Try Schedule-Free with higher base LR (paper recommends ~1.5-2x for Schedule-Free vs scheduled)
- Try Schedule-Free for ALL groups including Muon (replace warmdown entirely)
- Try different β₁ for Schedule-Free groups (0.9 instead of 0.8 for smoother averaging)
- Combine with intermediate RoPE range or VE LR tuning

## If it fails

- Polyak averaging doesn't work well in ~830 steps. Mark Schedule-Free as dead for short training.
- Try FP8 matmuls on Blackwell (2x throughput = 2x more steps)
- Try intermediate RoPE (1500, 8000, 60000, 300000)
- Try VE_LR=0.4


## Résultats
- val_bpb: 1.087675
- peak_vram_mb: 10534.7
- status: discarded
