Switch lm_head from AdamW to Muon optimizer for Newton-Schulz orthogonalized updates on the largest parameter matrix.

## What changed
- `lm_head` parameters moved from AdamW optimizer group to Muon optimizer group
- Removed the separate AdamW group for lm_head (was: LR=0.004, AdamW)
- lm_head now joins transformer matrix params in Muon groups (LR=0.04, NS orthogonalization)
- lm_head gets its own Muon group (unique shape 32768x512, no other params share it)
- lm_head now also gets cautious weight decay (was 0.0, now matches Muon WD=0.2 decayed)
- 3 lines changed in `setup_optimizer`

## Why
- **The lm_head is the single largest matrix** — 16.8M params, 44% of all matrix FLOPs. Its convergence quality directly impacts every prediction.
- **Newton-Schulz orthogonalization** pushes the 32768 output vectors toward orthogonality in 512-dim space. This maximizes discriminative power between tokens — the model can better separate similar tokens in the output distribution.
- **Currently AdamW with LR=0.004 is very conservative.** Muon's NS normalizes gradient directions and NorMuon handles step sizing adaptively. The lm_head should converge faster and to a better optimum.
- **NS compute is cheap for this shape.** The tall matrix branch computes `X.mT @ X` = 512x512 matmuls (5 iterations = ~0.25ms total). Negligible overhead.
- **This is the kind of "component replacement" the instructions prioritize**: swapping the optimization algorithm for a major component, not hyperparameter tuning.

## LR analysis
- For shape (32768, 512): Muon's internal LR scaling = `max(1, 32768/512)^0.5 = 8.0`
- Effective LR = 0.04 * 8.0 = 0.32 (vs AdamW's ~0.005)
- This sounds high, but NorMuon's second-momentum adaptive scaling self-corrects the step size
- Softcap=15 + z-loss provide safety bounds on logit magnitudes
- If the LR is too aggressive, NorMuon will grow its second_momentum buffer and reduce steps

## Weight decay note
- lm_head previously had WD=0.0 (AdamW group), now gets cautious WD=0.2 (Muon, decayed)
- Cautious WD only decays params where gradient and param agree in sign → conservative
- lm_head starts with tiny weights (std=0.001), so WD has minimal effect initially
- As weights grow during training, WD provides mild regularization

## Expected outcome
- If Muon's NS orthogonalization helps lm_head convergence: improvement of ~0.002-0.008
- If the effective LR (0.32) is too aggressive: regression of ~0.003-0.010 (NorMuon should self-correct, but the initial steps might overshoot)
- If Muon and AdamW are equivalent for this task: neutral within noise (~0.003)
- No crash risk — Muon handles rectangular matrices natively (tall matrix branch)
- No OOM risk — momentum buffer is (1, 32768, 512) = 64MB in BF16, second momentum is (1, 32768, 1) = 128KB
