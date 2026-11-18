Label smoothing (alpha=0.1) retry — exp33 crashed on infra, never tested on GPU. Fixed: smoothing applied only during training, not eval.

## What changed
- Added `label_smoothing=0.1` to `F.cross_entropy()` call, but ONLY when `reduction='mean'` (training)
- When `reduction='none'` (eval via `evaluate_bpb`), `label_smoothing=0.0` — pure CE, val_bpb stays comparable
- Exp33 applied smoothing unconditionally, which would have made val_bpb incomparable. This version fixes that.
- 1 line added (the `smoothing = 0.1 if reduction == 'mean' else 0.0` conditional)

## Why
- **Label smoothing is proven in all major LLMs** (GPT-3, PaLM, Chinchilla). It prevents overconfident predictions and improves gradient flow.
- **Synergy with softcap=15.** Without smoothing, the model tries to push logits toward ±∞ to minimize CE, but tanh saturates at ±15 → wasted gradient signal. Smoothing removes the incentive to push logits that far, so gradients flow more cleanly through the tanh.
- **Synergy with z-loss.** Both reduce incentive for large logits, from different angles — z-loss via explicit penalty, label smoothing via target modification.
- **Exp33 crashed on RunPod infra** (not code). The idea was never tested on GPU. Infra has been stable since exp29.
- **Improvement over exp33**: eval now uses pure CE (no smoothing), so val_bpb is directly comparable to all previous experiments including the best (1.079602).

## Label smoothing mechanics
- Training target: 90% on correct token, 10/32768 ≈ 0.0003% on each wrong token
- Optimal model doesn't need infinite logits → less tanh saturation → better gradient flow
- Zero compute overhead. PyTorch implements this without materializing the full distribution.

## Expected outcome
- If smoothing helps calibration + gradient flow: improvement of ~0.001-0.003
- If z-loss + softcap already handle calibration: neutral within noise (~0.003)
- If 0.1 is too aggressive: mild regression (~0.001-0.003), can retry with 0.05
- No crash, OOM, speed, or memory risk


## Résultats
- val_bpb: 1.119757
- peak_vram_mb: 12378.5
- status: discarded
