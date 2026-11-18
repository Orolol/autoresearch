Constant weight decay — remove linear WD schedule decay to prevent double-decay with LR warmdown.

## What changed
One-line change in `get_weight_decay()`: removed `* (1 - progress)` so WD stays at 0.2 throughout training instead of linearly decaying to 0.

## Why
The current WD schedule creates a double-decay: WD decays explicitly via `(1 - progress)` AND implicitly via the decaying LR (since the Muon update applies `lr * wd * params`). With WARMDOWN_RATIO=0.7, this means effective regularization drops quadratically — at progress=0.9, effective WD is only 0.003 instead of 0.028 with constant WD. This leaves the last ~50% of training with almost no regularization, potentially allowing weight magnitude drift during the critical convergence phase.

Standard LLM training (GPT-3, Chinchilla, etc.) uses constant WD and lets the LR schedule handle decay. The explicit WD decay is unusual and may be suboptimal.

## Expected outcome
- Best case: -0.003 (better regularization during warmdown prevents overfitting)
- Worst case: +0.005 (over-regularization prevents convergence)
- Most likely: neutral to slight improvement

## Next steps if this works
- Try intermediate WD values (0.1, 0.3) with constant schedule
- Try WD that increases during warmdown (inverse decay)

## Next steps if this fails
- WD schedule tuning is dead — current double-decay is fine
- Consider FP8 retry with different approach (simpler kernel, MLP-only)
- Consider novel loss function modifications (token-difficulty weighting)


## Résultats
- val_bpb: 1.067816
- peak_vram_mb: 11984.2
- status: discarded
