Interleaved local/global attention pattern — [G,L,G,L,L,G,L,G] instead of contiguous [G,G,L,L,L,L,G,G] for global context refresh between local layers.

## What changed

In `train.py`, changed one line in `CausalSelfAttention.__init__` to redistribute local/global attention layers:

- Old: `self.local_attn = (layer_idx >= 2 and layer_idx < n - 2)` → layers 2,3,4,5 local (contiguous block)
- New: `self.local_attn = (min(layer_idx, n - 1 - layer_idx) % 2 == 1)` → layers 1,3,4,6 local (interleaved)

Pattern comparison (depth=8):
- Before: [G, G, L, L, L, L, G, G] — 4 consecutive local layers in the middle
- After:  [G, L, G, L, L, G, L, G] — local layers interleaved with global refreshes

Same count: 4 local + 4 global layers. Zero new parameters. Zero compute change.

U-Net skip alignment: 0(G)→7(G), 1(L)→6(L), 2(G)→5(G), 3(L)→4(L) — perfectly symmetric type matching.

## Why

1. **Information flow bottleneck**: With 4 consecutive local layers (2-5), tokens only see a 256-token chunk + decayed recurrent state for 4 layers straight. Fine-grained global context degrades through this span. A global layer at position 2 and 5 "refreshes" full-sequence attention mid-stream.

2. **Structural change on new axis**: This is not a chunk attention modification (exhausted, 8 failures) or MLP distribution change (dead, 5 experiments). It's about the ARRANGEMENT of existing mechanisms — a new axis never explored.

3. **Inspired by interleaved architectures**: Many successful models (Gemma 2, Jamba) interleave different attention mechanisms rather than grouping them contiguously. The alternation creates a "breathe" pattern — focus locally, then contextualize globally.

4. **U-Net symmetry**: The new pattern creates type-matched skip connections (G→G, L→L), which may improve skip connection utility since same-type layers share more similar representational structure.

5. **Minimal risk**: One line change, no new params, no new ops, exact same compute budget. Compile-friendly (same layer types, same parameter shapes).

## Risks

- **Regression (medium)**: Layer 1 becoming local means less global context early in processing. The first global "refresh" doesn't happen until layer 2 (vs layer 0 currently getting 2 global layers before any local). However, layer 0 is still global and provides initial full-sequence context.
- **Crash (very low)**: No new ops, no shape changes, fully compile-compatible.
- **Neutral (medium)**: If 4 consecutive local layers already build sufficient context via recurrence, interleaving global refreshes might be redundant.
