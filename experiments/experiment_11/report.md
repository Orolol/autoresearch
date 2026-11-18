Embedding tying (wte = lm_head weight sharing) to save 16.8M params (22% of total model) and simplify the architecture.

## What changed
- `self.lm_head.weight = self.transformer.wte.weight` — output projection shares weights with token embedding
- Single initialization with `std=0.5` (compromise: original wte used 1.0, lm_head used 0.001)
- Removed lm_head as a separate optimizer group — shared weight uses `tied_lr = sqrt(embedding_lr * unembedding_lr) ≈ 0.049`
- Value embeddings keep their original LR (0.6)
- Updated parameter counting to reflect the tie (lm_head=0 unique params)

## Why
- **22% param reduction**: lm_head is 16.8M params (32768 × 512). With tying, these are shared with wte, so zero additional params.
- **Proven technique**: Weight tying is standard in GPT-2, T5, BERT, and early LLaMA. The embedding and output projection naturally want to be similar (tokens nearby in embedding space should have similar output probabilities).
- **Better convergence potential**: Fewer unique parameters to learn in 5 minutes. The output gradient provides additional signal to the embedding (and vice versa), enriching the gradient per step.
- **Safe with softcap**: Initial logits with std=0.5 embedding give logit std ≈ sqrt(512) × 0.5 ≈ 11.3. Softcap=15 clips gracefully (tanh(11.3/15)≈0.65, not saturating). Z-loss provides smooth gradient to push logits into good range.
- **AdamW handles merged gradients**: Per-element adaptive step sizes automatically balance the dense output gradient and sparse embedding gradient.

## Key design decision: compromise LR
- Original: embedding_lr=0.6, unembedding_lr=0.004 (150x apart)
- Tied weight uses geometric mean: `sqrt(0.6 × 0.004) ≈ 0.049`
- This is ~12x lower than the original embedding LR and ~12x higher than the original unembedding LR
- Value embeddings are unaffected (still at 0.6)

## Expected outcome
- If the param reduction and shared gradient signal help convergence: improvement of ~0.001-0.005
- If the compromise LR is wrong (too low for embedding or too high for output): regression of ~0.003-0.010
- If the effects balance out: neutral within noise (~0.003)
- No crash or OOM risk — pure simplification
