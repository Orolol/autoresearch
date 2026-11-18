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

## Config actuelle (best = 1.072821, exp90)

DEPTH=8, ASPECT_RATIO=64, HEAD_DIM=128 (4 heads), MLP 3x ReluSquared, softcap=13, z-loss=1e-4, TOTAL_BATCH_SIZE=2^18, DEVICE_BATCH_SIZE=32, EMBEDDING_LR=0.6, MATRIX_LR=0.04, WEIGHT_DECAY=0.2, WARMDOWN_RATIO=0.6, multi-scale RoPE (2500,10000,40000,160000), U-Net skip connections, VE bookend (first+last layer, shared weights, separate gates), x0_lambdas.

## Leçons confirmées (95 expériences)

### Ce qui a marché (gardé dans le baseline)
- **MLP 3x** (exp14, -0.008) : réduction compute → plus de steps. Le gain le plus important.
- **Softcap 13** (exp45, -0.001) : meilleur gradient flow à travers tanh.
- **Z-loss 1e-4** (exp9) : stabilise les logits. 1e-4 est l'optimum exact (5e-5 pire, 2e-4 pire).
- **U-Net skip connections** (exp71, -0.001) : skip_lambdas init=0.0, learned via AdamW.
- **Multi-scale RoPE** (exp80, -0.0002) : per-head bases (2500,10000,40000,160000).
- **VE bookend** (exp84, -0.004) : VE sur first+last layer, shared weights, separate gates.

### DEPTH=9 — jamais testé avec succès
7 tentatives (exp86-95), toutes crashées. Exp86/87/88 = bug IndexError dans U-Net skip pour odd depths. Exp92-95 = infra RunPod (CUDA driver init failure). Le bug est corrigé dans le code actuel (`if i >= half and mirror < half:`). Back-of-envelope : ~1.070-1.071 si ça marche. Config : DEPTH=9, ASPECT_RATIO=56 (→512 dim, 4 heads).

## Idées mortes (NE PAS retenter)

**Architecture** : parallel attn+MLP (+0.034), HEAD_DIM=64 (+0.012), factored lm_head (+0.016), DEPTH=7 (+0.033), MLP 2.5x (+0.003), pre-output MLP (+0.003), mid-MLP norm (+0.002), qk_scales (+0.002), VE all layers (+0.009)

**Activations** : SwiGLU (+0.004), GeluSquared (+0.006). ReluSquared est le meilleur à cette échelle.

**Optimizer** : Muon momentum warmup 300→100 (+0.0007), Muon beta2=0.99 (+0.001), momentum ceiling 0.97 (+0.004), lm_head Muon (2x crash), MATRIX_LR=0.06 (neutral), gradient clipping (+0.010), decouple AdamW/Muon LR (+0.013). **Muon est parfaitement tuné.**

**Training** : EMA (+0.016), label smoothing (+0.040 catastrophique), cosine warmdown (+0.004), WARMUP_RATIO=0.01 (+0.004), WEIGHT_DECAY=0.0 (+0.004), DEVICE_BATCH_SIZE=64 (+0.016), TOTAL_BATCH_SIZE=2^17 (+0.011) et 196608 (+0.012), torch.compile modes (tous pires ou crash), embedding tying (3x crash)

**Régularisation** : softcap=30 (+0.005), softcap=11 (+0.001), z-loss 5e-5 (+0.002), z-loss 2e-4 (+0.001), remove x0_lambdas (+0.020), remove VE (+0.023)

**Positional** : RoPE base=1000 (+0.013)

**Batch size** : 2^18 est l'optimum exact. Ne pas changer.

## Pistes restantes (non testées ou non concluantes)

1. **DEPTH=9** — jamais réellement testé sur GPU (7 crashs infra/bug). Potentiel ~-0.002. Haute priorité.
2. **DeltaNet / GLA** — remplacer softmax attention. Bold, complexe, potentiel élevé.
3. **Schedule-Free Adam** pour les groupes AdamW uniquement (garder Muon pour matrices). Complexe.
4. **FP8 matmuls** sur Blackwell — 2x throughput = 2x plus de steps. Dernier levier "more steps". Complexe.
5. **Wider multi-scale RoPE range** — (1000, 5000, 50000, 500000) pour plus de spécialisation.
6. **skip_lambdas init=0.05** — (exp82 crashed infra, jamais testé). Init plus proche de la zone utile.
7. **Mixture of Depths** — skip dynamique de layers. Torch.compile compat unclear.

## Bug fix documenté

U-Net skip pour odd depths : `if i >= half:` → `if i >= half and mirror < half:`. Sans ce fix, DEPTH=9 crash avec IndexError (mirror=4, skip_lambdas size=4, index out of range).

## Note sur MoE
torch.compile(dynamic=False) est incompatible avec sparse MoE routing (token counts dynamiques). Le seul MoE compile-friendly calcule TOUS les experts pour TOUS les tokens = double le compute MLP. MoE est effectivement mort sauf trick compile-friendly.
