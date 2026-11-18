Pre-output MLP — add a lightweight feedforward layer (2x expansion, ReluSquared) before lm_head for extra non-linear depth at ~3% compute cost.

## What changed

Added two new linear layers (`output_fc`: 512->1024, `output_proj`: 1024->512) with ReluSquared activation between the final transformer block and lm_head. The forward pass now does:

```
x = x + output_proj(relu(output_fc(norm(x))).square())
x = norm(x)
logits = lm_head(x)
```

Instead of:
```
x = norm(x)
logits = lm_head(x)
```

output_proj is zero-initialized so the model starts identical to baseline (MLP contribution is zero at step 0). output_fc uses the same uniform init as other matrix params. Both are optimized with Muon.

+1,048,576 parameters (1M), ~3% compute overhead per step.

## Why

DEPTH is the single most valuable axis in this series:
- exp18 (DEPTH=7): +0.033 catastrophe (losing 1 layer was devastating)
- exp22 (DEPTH=9 iso-compute): +0.001 (the 9th full layer's compute cost slightly exceeded its capacity benefit)

The output head maps 512-dim hidden states to 32768 vocab entries. A non-linear transformation before this projection can help the model form better output representations. This is essentially a "cheap 9th layer" — it adds the MLP capacity of a full layer (at 2/3 the hidden dim) without any attention overhead (which is the expensive part).

The ~3% compute overhead means ~25 fewer training steps out of ~830 total. The capacity benefit should outweigh this small step count reduction.

## Risks

- Zero crash/OOM risk (standard nn.Linear + ReluSquared, torch.compile compatible)
- If compute overhead > capacity benefit: mild regression ~0.001-0.002
- If pre-output processing is redundant with the last block: neutral result
- New Muon groups for the two new matrix shapes (1024x512, 512x1024) — single-param groups, but Muon handles this fine

## If it works

- Try 3x expansion (1536 hidden) for more capacity at ~4.5% overhead
- Try adding a second pre-output MLP (stacked) for even more depth
- Combine with z-loss 2e-4 (directional improvement confirmed)
- Try making the output MLP gated (ReGLU style) for better capacity/compute ratio

## If it fails

- The last transformer block already produces good enough representations for lm_head
- Try skip_lambdas init 0.1 instead of 0.0 for faster U-Net learning
- Try Differential Transformer (attention with subtraction between head pairs)
- Try increasing VE to first + last layer (more value embedding coverage)


## Résultats
- val_bpb: 1.080138
- peak_vram_mb: 10749.9
- status: discarded
