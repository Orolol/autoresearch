U-Net skip connections — connect early layer outputs to mirror late layers (0→7, 1→6, 2→5, 3→4) with learnable per-pair scaling.

## What changed

Added `self.skip_lambdas = nn.Parameter(torch.zeros(config.n_layer // 2))` to GPT (4 scalars for 8 layers). In the forward pass, layers 0-3 save their output after the block; layers 4-7 receive a skip connection from their mirror layer (layer 4 gets layer 3's output, layer 5 gets layer 2's, etc.), scaled by the corresponding skip_lambda.

The skip is applied BEFORE the block processes, alongside the existing resid_lambdas and x0_lambdas scaling:
```
x = resid_lambdas[i] * x + x0_lambdas[i] * x0 + skip_lambdas[mirror] * skip_states[mirror]
```

skip_lambdas are optimized with AdamW at scalar_lr=0.5, betas=(0.96, 0.95). Initialized to 0.0 (identical to baseline at step 0).

## Why

The model has two information pathways: the residual stream (cumulative processing) and x0 skip connections (raw embeddings). U-Net skips add a third: "diagonal" connections carrying partially-processed features. Layer 7 gets access to layer 0's output (1 block of processing), providing mid-level representations.

Benefits:
1. **Gradient flow**: Gradients from late layers can reach early layers through shorter paths (skip connection + 1 block), supplementing the long residual chain
2. **Multi-scale features**: Late layers see both highly-processed (residual) and lightly-processed (skip) representations
3. **Symmetry-aware**: Mirror pairing (0↔7, 1↔6) is well-motivated — early layers extract low-level features, late layers refine them, and the skip gives late layers direct access to the corresponding early-level features

This architecture pattern is used in DiT (Diffusion Transformer), Hourglass Transformers, and recent NanoGPT speedrun entries.

## Risks

- Zero crash/OOM risk (4 extra scalars, skip states already retained by autograd)
- Zero speed impact (4 scalar-weighted additions per forward pass)
- If existing residual + x0 pathways already provide sufficient gradient flow: neutral within noise
- If skip connections interfere with the tuned residual dynamics: mild regression ~0.001-0.003
- Possible torch.compile issue with list-based skip state storage (mitigated by using fixed-size list with static control flow)

## If it works

- Try initializing skip_lambdas to 0.1 instead of 0.0 for faster learning
- Try asymmetric skip: only connect first 2 layers to last 2 (skip_lambdas for pairs 0↔7 and 1↔6 only)
- Combine with z-loss coefficient tuning

## If it fails

- The residual + x0 pathways are sufficient. Cross-layer shortcuts don't help at this scale.
- Try DeltaNet or Gated Linear Attention (replacing softmax attention entirely)
- Try simple 2-expert MoE in MLP layers


## Résultats
- val_bpb: 1.077355
- peak_vram_mb: 10337.7
- status: IMPROVED
