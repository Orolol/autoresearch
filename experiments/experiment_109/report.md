Wider multi-scale RoPE range — (1000, 5000, 50000, 500000) for greater head specialization across distance scales.

## What changed

1-line change in `train.py` (line 492):

Before: `ROPE_BASES = (2500, 10000, 40000, 160000)` (geometric r=4, 64x range)
After: `ROPE_BASES = (1000, 5000, 50000, 500000)` (wider spread, 500x range)

## Why

- **Multi-scale RoPE already works** (exp80, -0.0002): per-head base frequencies improved the baseline. Widening the range pushes head specialization further.
- **Head specialization at 4 heads**: With only 4 attention heads, each one carries significant representational burden. Wider spread means each head covers a truly distinct distance scale:
  - Head 0 (base=1000): ultra-local, ~6 token period at fastest frequency
  - Head 1 (base=5000): medium-local, ~31 token period
  - Head 2 (base=50000): medium-range, ~314 token period
  - Head 3 (base=500000): near-position-agnostic, barely rotates across 2048 context
- **Content-only head**: base=500000 effectively provides a position-agnostic attention head that attends purely based on feature similarity — complementary to position-aware heads.
- **base=1000 was only bad uniformly**: exp65 used base=1000 for ALL heads (+0.013). Here only 1 of 4 heads uses it, while the others cover longer ranges.
- **Zero risk**: 1-line hyperparameter change, no new code, no new params, no OOM risk.

## Risks

- base=1000 might make head 0 too myopic (only attending to ~6 nearest tokens), wasting 25% of attention capacity
- base=500000 might make head 3 completely position-blind, reducing it to a bag-of-words attention
- The optimal spread might be narrower — current 64x could already be near-optimal
- Expected range: -0.003 to +0.005

## If it works

- Try even wider: (500, 5000, 100000, 2000000)
- Try asymmetric spread: (2000, 5000, 50000, 1000000) — keep the middle similar, push extremes
- Try 5 heads (HEAD_DIM=96 or 102) with more granular RoPE bases
- Combine with VE LR tuning or Schedule-Free Adam

## If it fails

- The current 64x range is already near-optimal. Mark wider RoPE as dead.
- Try Schedule-Free Adam for AdamW groups (option 2)
- Try FP8 matmuls on Blackwell (option 3)
- Try VE betas tuning (option 13)


## Résultats
- val_bpb: 1.070391
- peak_vram_mb: 10450.8
- status: discarded
