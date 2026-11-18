Mamba-style causal depthwise convolution (kernel=4) on Q/K projections in local linear attention layers (2-5) for local position mixing before chunkwise linear attention.

## What changed

In `train.py`, added short causal depthwise conv1d (kernel_size=4) on Q and K projections in local attention layers (2-5), applied AFTER linear projection but BEFORE reshape into heads, RoPE, and QK-norm:

1. **New parameters** (`CausalSelfAttention.__init__`, local layers only):
   - `q_conv`: Conv1d(n_embd, n_embd, kernel_size=4, padding=3, groups=n_embd, bias=False)
   - `k_conv`: Conv1d(n_embd, n_embd, kernel_size=4, padding=3, groups=n_embd, bias=False)
   - Total: 4 layers x 2 convs x (512 channels x 4 kernel) = 16,384 new parameters (~0.02% of model)

2. **Modified forward** (`CausalSelfAttention.forward`):
   - Q/K projections are computed as before, but for local_attn layers, a causal depthwise conv is applied before reshaping into heads
   - `q = q_conv(q.T)[:, :, :T].T` — causal conv with right-padding truncated
   - Same for K

3. **Init** (`GPT.init_weights`):
   - Conv weights initialized to identity: `weight[:, :, -1] = 1.0`, all others 0
   - At step 0, the conv passes through the input unchanged → exact baseline equivalence

## Why

1. **Priority queue directive**: "try short causal conv (kernel=4) on Q/K before linear attention (Mamba-style preprocessing)" is explicitly listed as a next step after DeltaNet/GLA variations.
2. **Mamba architecture inspiration**: Mamba uses a causal conv before the SSM to inject local position information. Our linear attention is analogous to an SSM — it lacks softmax's implicit local structure detection.
3. **Complementary to token shift**: Token shift mixes current+previous token (2-token window) on the raw input. The conv operates on Q/K projections with a 4-token window — different space, different window, different purpose.
4. **Compensates for no softmax**: Softmax attention implicitly captures local position structure through its normalization. Linear attention with decay doesn't have this — the causal conv adds it back.
5. **Identity at init**: Zero regression risk at step 0. The model starts identical to exp179 baseline.

## Risks

- **Regression (low-medium)**: The conv adds compute (~3% overhead) which could reduce step count slightly. But the identity init means no regression at step 0.
- **Crash (very low)**: Conv1d is a standard torch op, fully compile-friendly. Fixed sequence length = no dynamic shapes.
- **Compile time (low)**: Conv1d is well-optimized in torch.compile. No new control flow.
- **Neutral (medium)**: If the linear attention already captures enough local structure via the decay mechanism, the conv may not add useful information.


## Résultats (after crash fix)
- val_bpb: 1.056455
- peak_vram_mb: 14632.8
- status: discarded
