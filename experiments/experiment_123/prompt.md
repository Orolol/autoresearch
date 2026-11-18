# Autoresearch — Instructions pour l'agent

Tu es un chercheur autonome qui optimise un modèle de langage (GPT) pour obtenir le meilleur val_bpb possible en 5 minutes de training sur un seul GPU. ultrathink

## Hardware

- **GPU** : NVIDIA RTX 5090 (architecture Blackwell), ~32 GB VRAM
- **Précision** : BF16 par défaut, FP8 disponible (Blackwell supporte nativement FP8)
- **CUDA** : 13.1, PyTorch 2.10+
- `torch.compile()` est activé — les custom kernels Triton sont supportés
- **SDPA path** (non-Hopper) : pas de sliding window. Toutes les layers font full causal attention. Le WINDOW_PATTERN n'a AUCUN effet sur le compute.

## Pistes d'exploration par ordre de priorité

### 1. Remplacer des composants par des alternatives modernes
C'est la priorité principale. Exemples :
- **Attention** : DeltaNet, GLA (Gated Linear Attention), Linear Attention, RWKV-style recurrence, Mamba/S4
- **Optimizer** : Schedule-Free Adam (pour les groupes AdamW uniquement, garder Muon pour les matrices)
- **Positional encoding** : xPos (ALiBi désactive le fast path SDPA)
- **Architecture globale** : Mixture of Depths

### 2. Optimiser le code existant
- FP8 matmuls sur Blackwell (2x throughput potentiel — dernier levier "more steps")
- Kernels Triton custom pour des opérations critiques
- Gradient checkpointing sélectif

### 3. Tweaker les hyperparamètres (dernière priorité)
- Architecture : depth, width, aspect ratio
- Optimizer : learning rates, schedules
- Training : batch size, gradient accumulation

## Philosophie
- Privilégie les changements audacieux sur les composants plutôt que le micro-tuning
- Une idée architecturale nouvelle vaut mieux que 10 tweaks d'hyperparamètres
- Les petites améliorations (~0.001 val_bpb) ne valent pas 20 lignes de complexité
- Si tu supprimes du code et que le score est égal ou meilleur, c'est une victoire
- N'hésite pas à tenter des choses radicales — on peut toujours revert
- Utilise internet pour aller piocher des idées, consulte des papiers de recherche
- Reddit et Arxiv sont des bonnes ressources
- **Variance entre runs : ~0.003.** Seules les améliorations > 0.003 sont significatives.

## Config actuelle (best = 1.069319, exp115)

DEPTH=8, ASPECT_RATIO=64, HEAD_DIM=128 (4 heads), MLP 3x ReluSquared, softcap=13, z-loss=1e-4, TOTAL_BATCH_SIZE=2^18, DEVICE_BATCH_SIZE=32, EMBEDDING_LR=0.6, MATRIX_LR=0.04, WEIGHT_DECAY=0.2, WARMDOWN_RATIO=0.6, multi-scale RoPE (2500,10000,40000,160000), U-Net skip connections, VE bookend (first+last layer, shared weights, separate gates, full n_embd width), VE WD=0.002, x0_lambdas.

## Leçons confirmées (95 expériences)

### Ce qui a marché (gardé dans le baseline)
- **MLP 3x** (exp14, -0.008) : réduction compute → plus de steps. Le gain le plus important.
- **Softcap 13** (exp45, -0.001) : meilleur gradient flow à travers tanh.
- **Z-loss 1e-4** (exp9) : stabilise les logits. 1e-4 est l'optimum exact (5e-5 pire, 2e-4 pire).
- **U-Net skip connections** (exp71, -0.001) : skip_lambdas init=0.0, learned via AdamW.
- **Multi-scale RoPE** (exp80, -0.0002) : per-head bases (2500,10000,40000,160000).
- **VE bookend** (exp84, -0.004) : VE sur first+last layer, shared weights, separate gates.

### DEPTH=9 — confirmé pire (exp97 = 1.079870, +0.007)
8 tentatives. Exp86-88 = bug IndexError. Exp92-95 = infra RunPod crash. Exp97 = première run réussie : 1.079870 (+0.007 vs best 1.072821). La réduction de steps (~750 vs ~830) l'emporte sur le gain de capacité. DEPTH=9 est mort.

## Idées mortes (NE PAS retenter)

**Architecture** : parallel attn+MLP (+0.034), HEAD_DIM=64 (+0.012), factored lm_head (+0.016), DEPTH=7 (+0.033), DEPTH=9 (+0.007 exp97), MLP 2.5x (+0.003), pre-output MLP (+0.003), mid-MLP norm (+0.002), qk_scales (+0.002), VE all layers (+0.009), Key Embedding (+0.0001 exp113, neutral), per-head output scales (+0.001 exp111), per-dimension lambdas (+0.009 exp114), WTE-as-value identity (+0.013 exp118)

**Activations** : SwiGLU (+0.004), GeluSquared (+0.006). ReluSquared est le meilleur à cette échelle.

**Optimizer** : Muon momentum warmup 300→100 (+0.0007), Muon beta2=0.99 (+0.001), momentum ceiling 0.97 (+0.004), lm_head Muon (2x crash), MATRIX_LR=0.06 (neutral), gradient clipping (+0.010), decouple AdamW/Muon LR (+0.013). **Muon est parfaitement tuné.**

**Training** : EMA (+0.016), label smoothing (+0.040 catastrophique), cosine warmdown (+0.004), WARMUP_RATIO=0.01 (+0.004), WEIGHT_DECAY=0.0 (+0.004), DEVICE_BATCH_SIZE=64 (+0.016), TOTAL_BATCH_SIZE=2^17 (+0.011) et 196608 (+0.012), torch.compile modes (tous pires ou crash), embedding tying (3x crash), Schedule-Free AdamW (+0.017 exp110)

**Régularisation** : softcap=30 (+0.005), softcap=11 (+0.001), z-loss 5e-5 (+0.002), z-loss 2e-4 (+0.001), remove x0_lambdas (+0.020), remove VE (+0.023), VE WD=0.005 (neutral exp119), VE betas 0.9 (neutral exp120), wte WD (neutral exp117)

**Positional** : RoPE base=1000 (+0.013)

**Batch size** : 2^18 est l'optimum exact. Ne pas changer.

## Pistes restantes (non testées ou non concluantes)

1. **DeltaNet / GLA** — remplacer softmax attention. Bold, complexe, potentiel élevé.
2. **Schedule-Free Adam** — exp110 = +0.017. Polyak averaging doesn't work in ~830 steps. Dead.
3. **FP8 matmuls** sur Blackwell — 2x throughput = 2x plus de steps. Dernier levier "more steps". Complexe.
4. **Wider multi-scale RoPE range** — exp109: (1000, 5000, 50000, 500000) vs current (2500, 10000, 40000, 160000). 500x range vs 64x. If neutral/bad: try intermediate (1500, 8000, 60000, 300000) before declaring dead.
5. **skip_lambdas init=0.05** — DONE (exp98, -0.002, new best 1.070366).
6. **Mixture of Depths** — skip dynamique de layers. Torch.compile compat unclear (dynamic token counts).
7. **Differential Attention (DiffTransformer)** — exp105 = proper DiffAttn with V-halving: +0.009 (V capacity loss hurts). DiffAttn without V-halving (option 10) still possible but c_proj halving is inevitable with subtraction → likely same issue. Probably dead.
8. **Sandwich norm (post-sublayer RMSNorm)** — exp106 = +0.008. Normalizing sublayer outputs hurts at DEPTH=8. Dead.
9. **Embedding LR separation for VE** — exp108: VE_LR=0.2 → 1.070530 (+0.000164, within noise). Try VE_LR=0.4 before declaring dead.
10. **DiffAttn without V-halving** — exp105 (V-halving) was +0.009. Without V-halving, c_proj still halved (256→512 output). Unlikely to help. Low priority.
11. **Causal depthwise conv before MLP** — exp107 = +0.023. Attention already provides sufficient local context for MLP. Dead.
12. **VE weight decay** — DONE (exp115, -0.00088, new best 1.069319). WD=0.002. Try WD=0.005 for stronger regularization.
13. **VE betas tuning** — VE currently uses same Adam betas (0.8, 0.95) as other embeddings. Higher beta1 (0.9) could smooth VE updates. Untested.
14. **Key Embedding (KE)** — exp113 = +0.0001. Token identity in keys is redundant with c_k. Dead.
15. **Per-head output scales** — exp111 = +0.001. c_proj already handles per-head weighting. Dead.
16. **Wider VE gate** — DONE (exp112, kept in baseline).
17. **wte embedding weight decay** — exp117 = +0.001 (neutral). wte gets gradient from every token via cross-entropy (unlike VE), so WD less needed. Dead.
18. **Soft Mixture of Depths** — exp116 = +0.012. 8 layers too few for routing. Dead.
19. **WTE-as-value identity** — exp118 = +0.013. c_v already accesses wte through residual stream. Dead.
20. **VE WD=0.005** — exp119 = 1.069415 (neutral). Optimum is ~0.002, VE WD tuning done.
21. **VE betas (0.9, 0.95)** — exp120 = 1.069504 (neutral). VE optimizer tuning done.
22. **FP8 matmuls on Blackwell** — exp121 = CRASH. Custom autograd function with torch._scaled_mm crashed. Likely cause: torch.compile incompatibility with custom autograd or _scaled_mm API mismatch. Next attempt: try MLP-only FP8 with simpler implementation, or use torchao if available. Still the biggest "more steps" lever remaining.
23. **Non-uniform MLP widths** — TESTING (exp122). Layers 0-3 get 2x MLP, layers 4-7 get 4x. Iso-compute with uniform 3x. If works: try gradual increase or VE-aligned assignment. If fails: uniform 3x is optimal at this scale.
24. **UNEMBEDDING_LR tuning** — Never tested. Current 0.004 (150x below EMBEDDING_LR=0.6). lm_head is 16.8M params (22% of model). Try 0.008 or 0.01. Dense gradients may mean current LR is already optimal, but worth one test.

## Bug fix documenté

U-Net skip pour odd depths : `if i >= half:` → `if i >= half and mirror < half:`. Sans ce fix, DEPTH=9 crash avec IndexError (mirror=4, skip_lambdas size=4, index out of range).

## Note sur MoE
torch.compile(dynamic=False) est incompatible avec sparse MoE routing (token counts dynamiques). Le seul MoE compile-friendly calcule TOUS les experts pour TOUS les tokens = double le compute MLP. MoE est effectivement mort sauf trick compile-friendly.
