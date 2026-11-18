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

## Config actuelle (best = 1.070366, exp98)

DEPTH=8, ASPECT_RATIO=64, HEAD_DIM=128 (4 heads), MLP 3x ReluSquared, softcap=13, z-loss=1e-4, TOTAL_BATCH_SIZE=2^18, DEVICE_BATCH_SIZE=32, EMBEDDING_LR=0.6, MATRIX_LR=0.04, WEIGHT_DECAY=0.2, WARMDOWN_RATIO=0.6, multi-scale RoPE (2500,10000,40000,160000), U-Net skip connections, VE bookend (first+last layer, shared weights, separate gates), x0_lambdas.

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

**Architecture** : parallel attn+MLP (+0.034), HEAD_DIM=64 (+0.012), factored lm_head (+0.016), DEPTH=7 (+0.033), DEPTH=9 (+0.007 exp97), MLP 2.5x (+0.003), pre-output MLP (+0.003), mid-MLP norm (+0.002), qk_scales (+0.002), VE all layers (+0.009)

**Activations** : SwiGLU (+0.004), GeluSquared (+0.006). ReluSquared est le meilleur à cette échelle.

**Optimizer** : Muon momentum warmup 300→100 (+0.0007), Muon beta2=0.99 (+0.001), momentum ceiling 0.97 (+0.004), lm_head Muon (2x crash), MATRIX_LR=0.06 (neutral), gradient clipping (+0.010), decouple AdamW/Muon LR (+0.013). **Muon est parfaitement tuné.**

**Training** : EMA (+0.016), label smoothing (+0.040 catastrophique), cosine warmdown (+0.004), WARMUP_RATIO=0.01 (+0.004), WEIGHT_DECAY=0.0 (+0.004), DEVICE_BATCH_SIZE=64 (+0.016), TOTAL_BATCH_SIZE=2^17 (+0.011) et 196608 (+0.012), torch.compile modes (tous pires ou crash), embedding tying (3x crash)

**Régularisation** : softcap=30 (+0.005), softcap=11 (+0.001), z-loss 5e-5 (+0.002), z-loss 2e-4 (+0.001), remove x0_lambdas (+0.020), remove VE (+0.023)

**Positional** : RoPE base=1000 (+0.013)

**Batch size** : 2^18 est l'optimum exact. Ne pas changer.

## Pistes restantes (non testées ou non concluantes)

1. **DeltaNet / GLA** — remplacer softmax attention. Bold, complexe, potentiel élevé.
2. **Schedule-Free Adam** pour les groupes AdamW uniquement (garder Muon pour matrices). Complexe.
3. **FP8 matmuls** sur Blackwell — 2x throughput = 2x plus de steps. Dernier levier "more steps". Complexe.
4. **Wider multi-scale RoPE range** — (1000, 5000, 50000, 500000) pour plus de spécialisation.
5. **skip_lambdas init=0.05** — DONE (exp98, -0.002, new best 1.070366).
6. **Mixture of Depths** — skip dynamique de layers. Torch.compile compat unclear (dynamic token counts).
7. **Differential Attention (DiffTransformer)** — en cours de test (exp99). Split Q/K/V into 2 sub-heads, SDPA with 2x heads at HEAD_DIM//2, subtract pairs. c_proj halved (n_embd//2 → n_embd). Fixed lambda=0.5. Si ça marche: try learnable lambda or lambda=0.8. Si ça rate: HEAD_DIM//2=64 trop petit, try doubling logical heads instead.
8. **Post-attention/post-MLP norm (sandwich norm)** — not tried. Different from mid-MLP norm (exp79, dead). Adds RMSNorm after c_proj output, before residual add. Could stabilize residual stream growth.
9. **Embedding LR separation for VE** — VE uses same LR as token embeddings (0.6). VE might benefit from lower LR since it regularizes values, not drives them.

## Bug fix documenté

U-Net skip pour odd depths : `if i >= half:` → `if i >= half and mirror < half:`. Sans ce fix, DEPTH=9 crash avec IndexError (mirror=4, skip_lambdas size=4, index out of range).

## Note sur MoE
torch.compile(dynamic=False) est incompatible avec sparse MoE routing (token counts dynamiques). Le seul MoE compile-friendly calcule TOUS les experts pour TOUS les tokens = double le compute MLP. MoE est effectivement mort sauf trick compile-friendly.
