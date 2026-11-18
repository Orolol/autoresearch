DEPTH=9 with U-Net skip bug fix — the fix was missing from the persisted train.py despite being documented.

## Changes
1. **DEPTH = 9** (was 8) — adds one transformer layer for deeper modeling capacity
2. **ASPECT_RATIO = 56** (was 64) — keeps model_dim=512 (9*56=504, rounded up to 512 for HEAD_DIM=128 alignment), 4 heads
3. **U-Net skip bug fix** (line 324): `if i >= half:` → `if i >= half and mirror < half:` — prevents IndexError when n_layer is odd. For DEPTH=9: half=4, the middle layer (i=4, mirror=4) would try to access skip_lambdas[4] which doesn't exist. The guard skips this layer correctly.

## Why
DEPTH=9 is the #1 remaining priority. It has never successfully run on GPU:
- Exp86/87/88: crashed due to the IndexError bug (skip_lambdas[4] out of range)
- Exp92-95: documented as having the fix, but the fix was NOT persisted to the main train.py — those experiments applied it at runtime but crashed on RunPod infra before completing

All accumulated gradient flow improvements (U-Net skips, VE bookends, multi-scale RoPE, softcap 13) specifically help deeper models. Exp22 tested DEPTH=9 with MLP 2.5x before these improvements and got 1.080504. With MLP 3x and all improvements, back-of-envelope estimate: ~1.070-1.071.

## Risks
- ~8-10% fewer training steps due to extra layer compute
- If step reduction outweighs capacity gain: regression ~0.001-0.003
- Zero OOM risk (model_dim 512 is same as before)


## Résultats
- val_bpb: 1.079870
- peak_vram_mb: 11579.8
- status: discarded
