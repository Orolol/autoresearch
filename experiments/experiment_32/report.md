Retry lm_head with Muon optimizer at conservative LR (exp23 crashed, likely infra — never tested on GPU).

## What changed
- Moved lm_head (32768×512, 16.8M params) from AdamW to a dedicated Muon optimizer group
- Base LR = 0.005 (vs exp23's 0.04) → effective LR = 0.005 × sqrt(32768/512) = 0.04
- This matches the effective LR of square transformer matrices (0.04 × 1 = 0.04)
- Exp23 used the default matrix_lr=0.04 → effective 0.32 (8× too high), which may have contributed to instability
- lm_head now gets cautious weight decay (was 0.0 with AdamW), decaying from 0.2 to 0 during training
- Removed lm_head from AdamW param groups
- ~4 lines changed in `setup_optimizer`

## Why
- **The lm_head is the single largest parameter matrix** (16.8M params, 44% of all matrix FLOPs). It's the biggest target for optimization improvements.
- **AdamW is fundamentally slow to start.** It needs many steps to warm up its per-element second moment estimates. With only ~830 training steps, AdamW for lm_head barely converges. Muon's Newton-Schulz orthogonalization provides *instant* gradient preconditioning — no warmup needed.
- **NS orthogonalization is a perfect structural prior for output projection.** It pushes 32768 output token vectors toward approximate isometry in 512-dim space, maximizing discriminative separation between tokens. This is exactly what an output layer needs.
- **Exp23 crashed** but was in the middle of a cluster of 4+ consecutive infra crashes (exp25-28 also all crashed). The crash was almost certainly RunPod infra, not a code bug.
- **The notes explicitly recommend** retrying with lower LR: "Retry with lm_head-specific lower LR (create separate Muon group for lm_head with lr=0.005, effective=0.04)."

## Muon mechanics for lm_head
- Shape (32768, 512): tall matrix, so NS uses `X.mT @ X` = 512×512 inner products (very cheap)
- 5 NS iterations = 5 × batched 512×512 matmuls ≈ 0.05ms (negligible)
- NorMuon variance reduction: second momentum buffer shape (1, 32768, 1) — tiny
- Momentum buffer: (1, 32768, 512) ≈ 33MB — same as AdamW's exp_avg (saves exp_avg_sq = 33MB)
- Net memory: ~33MB LESS than AdamW (no exp_avg_sq needed)

## LR analysis
- AdamW was: 0.004 × (512/768)^-0.5 = 0.0049 effective
- Muon: 0.005 base × 8.0 rectangular scaling = 0.04 effective
- 8× higher effective LR than AdamW, BUT Muon's step direction is pre-conditioned by NS (not raw gradient)
- NorMuon self-corrects step size via second momentum — if 0.04 is too aggressive, it adapts down

## Weight decay note
- lm_head now gets cautious WD (was 0.0 with AdamW). Cautious mask `(g * p) >= 0` filters updates, and WD decays from 0.2 to 0. Combined with softcap=15 limiting logit magnitudes, mild WD on output vectors should be fine.
- If result is close but slightly worse, isolating WD=0 for lm_head Muon is a follow-up experiment.

## Expected outcome
- If Muon's instant preconditioning helps lm_head convergence: improvement of ~0.001-0.005
- If AdamW was already fine for lm_head at this short duration: neutral within noise (~0.003)
- If effective LR 0.04 is too aggressive or cautious WD hurts: mild regression (~0.001-0.003)
- No crash risk (code verified, infra now stable)
- No OOM risk (~33MB less memory than AdamW)
