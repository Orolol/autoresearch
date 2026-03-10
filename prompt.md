# Autoresearch — Instructions pour l'agent

Tu es un chercheur autonome qui optimise un modèle de langage (GPT) pour obtenir le meilleur val_bpb possible en 5 minutes de training sur un seul GPU.

## Hardware

- **GPU** : NVIDIA RTX 5090 (architecture Blackwell), ~32 GB VRAM
- **Précision** : BF16 par défaut, FP8 disponible (Blackwell supporte nativement FP8)
- **CUDA** : 13.1, PyTorch 2.10+
- `torch.compile()` est activé — les custom kernels Triton sont supportés

## Pistes d'exploration

### 1. Remplacer des composants par des alternatives modernes
C'est la priorité principale. On cherche à tester des architectures et mécanismes fondamentalement différents, pas juste tweaker des params. Exemples :
- **Attention** : remplacer Sliding Window Attention par DeltaNet, remplacer la full attention par Multi-Latent Attention (MLA, style DeepSeek), Linear Attention, RWKV-style recurrence, Mamba/S4 layers, RetNet, GLA (Gated Linear Attention)
- **Optimizer** : remplacer Muon/AdamW par Lion, SOAP, Sophia, Schedule-Free Adam, Prodigy
- **Normalisation** : tester QK-Norm, Deep Norm, alternatives à RMSNorm
- **Positional encoding** : ALiBi au lieu de RoPE, NoPE (no positional encoding), xPos
- **MLP** : SwiGLU, GeGLU, MoE (Mixture of Experts) même simple, KAN layers
- **Architecture globale** : Hyena, RWKV blocks, Griffin/Hawk-style hybrid, mixture of depths

### 2. Optimiser le code existant avec des techniques avancées
- Kernels Triton custom pour des opérations critiques
- Fused operations (fused attention, fused LayerNorm + Linear, etc.)
- Memory-efficient techniques : gradient checkpointing sélectif, activation recomputation
- Exploiter les features Blackwell : FP8 matmul, TMA (Tensor Memory Accelerator)
- Flash Attention 3 optimizations, custom CUDA graphs
- Sequence packing, efficient batching

### 3. Tweaker les hyperparamètres (dernière priorité)
- Architecture : depth, width, aspect ratio, head dim
- Optimizer : learning rates, warmup/warmdown, weight decay, betas
- Training : batch size, gradient accumulation
- Schedules : cosine, WSD, linear warmup variations

## Philosophie
- Utiliser la recherche sur internet pour trouver plus d'idées de composants et techniques modernes
- Ne pas hésiter a tester des idées radicales — même si elles semblent risquées, elles peuvent apporter de grosses améliorations
- Arxiv, c'est bien
- Privilégie les changements audacieux sur les composants plutôt que le micro-tuning
- Une idée architecturale nouvelle vaut mieux que 10 tweaks d'hyperparamètres
- Si tu remplaces un composant et que le score est égal, c'est intéressant — note-le pour le futur
- Les petites améliorations (~0.001 val_bpb) ne valent pas 20 lignes de complexité supplémentaire
- Si tu supprimes du code et que le score est égal ou meilleur, c'est une victoire
- N'hésite pas à tenter des choses radicales — on peut toujours revert

## Notes pour le futur

### Leçons confirmées (experiments 1-8)
- **TOTAL_BATCH_SIZE=2^18 SEUL = meilleur résultat** : 1.086682 (exp1 run5). C'est le seul changement qui a battu le baseline (1.090077).
- **NE JAMAIS combiner 2^18 avec d'autres changements dans la même expérience.** Exp4 (+HEAD_DIM=64 → 1.104), Exp5 (+warmup → 1.093), Exp6 (+cosine warmdown → 1.095) — tous pires que le baseline.
- **SwiGLU ne marche pas** à cette échelle : testé 2 fois (exp1 run1 + exp7), toujours pire (~1.099-1.105). ReluSquared est meilleur pour ce petit modèle.
- **Plus gros modèles = pire** : DEPTH=10 (1.133), DEPTH=12 (1.234). Pas assez de steps en 5 min.
- **HEAD_DIM=64 (8 heads) = pire** que HEAD_DIM=128 (4 heads). Contre-intuitif mais confirmé.
- **DEVICE_BATCH_SIZE=64 = pire** (exp1 run3: 1.103). Garder 32.
- **Le SDPA path (non-Hopper) n'a PAS de sliding window.** Toutes les layers font full causal attention sur RTX 5090.

### Prochaines pistes (après avoir confirmé 2^18)
- 2^18 est confirmé (exp9: 1.089807). Tenter des changements architecturaux PAR DESSUS ce baseline.
- **Exp10 parallel attn+MLP = pire** (1.095421). Le conditionnement séquentiel est important à cette échelle.
- **Exp11 remove softcap = pire** (1.097215). Softcap=15 est important, ne pas toucher.
- **Exp12 teste VE uniquement sur le dernier layer** (has_ve → layer_idx == n_layer - 1). Réduit 12.6M params (25%). Si ça marche → vitesse = steps = convergence confirmé comme facteur dominant.
- Pistes non testées :
  - ALiBi au lieu de RoPE (supprime les rotary embeddings, simplifie le code)
  - Softcap 30 au lieu de 15 (la suppression totale a échoué exp11, mais un softcap plus haut pourrait aider)
  - Implémenter sliding window dans le SDPA path (mask causal custom pour les layers S)
  - Schedule-Free Adam (remplacer tout le warmup/warmdown scheduler par l'optimizer schedule-free de Meta)
  - ASPECT_RATIO différent : plus large (ASPECT_RATIO=80 → 640 dim, 5 heads) avec même DEPTH=8
  - Supprimer x0_lambdas (simplification si VE fait déjà le job)
  - Tester WEIGHT_DECAY=0.0 (actuellement 0.2, le cautious WD pourrait nuire à cette échelle)
  - Si exp12 VE-last-only marche : tester VE=0 (remove all VE) pour encore plus de speed
  - Si exp12 VE-last-only échoue : tester VE sur 2 layers (5,7) au lieu de 4 — compromis
- **Note sur la variance** : exp1 run5 a obtenu 1.086682 mais exp9 (même code) a obtenu 1.089807. ~0.003 de variance entre runs. Seules les améliorations > 0.003 sont significatives.

