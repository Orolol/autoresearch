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

### Leçons de la série actuelle (experiments 0-5)
- **Baseline = 1.088007** avec TOTAL_BATCH_SIZE=2^18, VE-last-only, softcap=15, x0_lambdas, ReluSquared MLP
- **VE=0 = 1.111** (+0.023). VE est critique, ne pas supprimer.
- **Softcap 30 = 1.093** (+0.005). Softcap=15 est bien calibré.
- **Remove x0_lambdas = 1.108** (+0.020). x0_lambdas sont importants pour le gradient flow.
- **GQA n_kv_head=2 = 1.088248** (+0.000241). Essentially tied with baseline. KV capacity reduction almost free at this scale. GQA is viable.
- **WEIGHT_DECAY=0.0 = 1.091936** (+0.004). WD=0.2 is better, keep it. Mild regularization helps even in short training.
- **Exp6 teste TOTAL_BATCH_SIZE=2^17** (doubler les optimizer steps, grad_accum=2). Si ça marche → "more steps" hypothesis extends further. Si ça échoue → 2^18 est le sweet spot.
- **Exp6 2^17 = pire** (1.098882, +0.011). 2^18 est le sweet spot confirmé. Les gradients deviennent trop bruiteux avec des batches plus petits.
- **Exp7 MATRIX_LR=0.06 = crash infra** (RunPod API timeout, code never ran on GPU). Re-tested in exp8.
- **Exp8 MATRIX_LR=0.06 = 1.088767** (+0.00076). Within noise — 0.04 is fine, 0.06 doesn't help.
- **Exp9 Z-loss = 1.087971** (-0.000036). Marginal improvement, within noise but kept. Z-loss helps gradient flow through softcap.
- **Exp10 teste MTP** (Multi-Token Prediction) : auxiliary loss predicting token i+2 via learned projection + shared lm_head. Weight 0.1. ~10-15% compute overhead.
- **Exp10 MTP crashed** — likely a code bug. Worth debugging and retrying if we need more ideas.
- **Exp11 teste embedding tying** (wte = lm_head weight sharing). LR compromise: geometric mean of 0.6 and 0.004 ≈ 0.049. Init std=0.5.
- Pistes non testées restantes :
  - Schedule-Free Adam (remplace tout le scheduler par un optimizer adaptatif, pourrait être le meilleur next step)
  - Differential Transformer (attention par paires de heads avec soustraction, mais complexe à implémenter avec SDPA)
  - MQA (n_kv_head=1) — GQA était un tie, MQA pourrait aller plus loin, mais VE dim shrinks to 128
  - Low-rank VE (réduire VE embedding dim avec projection, save ~87% VE params)
  - Stochastic Depth (layer drop) : skip aléatoire de layers pendant le training pour régulariser + accélérer (NOTE: impossible à accélérer avec torch.compile car branches = graph breaks, seul l'effet régularisation marche)
  - WARMDOWN_RATIO=0.4 (actuellement 0.6) : plus de steps à full LR, mais risque de sous-convergence en fin de training
  - Note: ALiBi désactive le fast path SDPA (custom mask nécessaire), probablement pas worth it
  - Si exp11 embedding tying marche : tester des LR plus agressives (0.1, 0.2) pour le tied weight
  - Si exp11 embedding tying échoue : le compromis LR est probablement le problème. Tester avec un LR plus proche de 0.004 (ex: 0.01) ou plus proche de 0.6 (ex: 0.3)
  - Si z-loss marche : tester coefficient plus grand (1e-3) ou plus petit (1e-5)
  - **MTP est mort** : exp12 a donné 1.093 (+0.005). Le compute overhead (~10-15% fewer steps) ne compense pas le gradient signal supplémentaire. Abandonner MTP.
  - Embedding tying (exp11 crashed) : à retenter proprement si on manque d'idées. Attention au compromis LR.
  - **Exp13 teste WARMDOWN_RATIO=0.4** (actuellement 0.6). Si ça marche → tester 0.3 (encore plus agressif). Si ça échoue → 0.6 est optimal.
  - Si WARMDOWN_RATIO=0.4 marche : combiner avec z-loss coefficient tuning (1e-3 ou 1e-5)
  - **Exp13 WARMDOWN_RATIO=0.4 = 1.089407** (+0.0014). Within noise but no improvement. 0.6 is fine.
  - **Exp14 teste MLP 3x** (hidden dim 1536 au lieu de 2048). Saves ~4.2M params, faster fwd/bwd → more steps.
  - Prochaines pistes si exp14 échoue :
    - MLP 2x (plus agressif, 1024 hidden dim — gros gain de vitesse mais capacité très réduite)
    - `torch.compile(mode="max-autotune")` au lieu de default — autotuned kernels + CUDA graphs potentiels, zero architectural change
    - WARMDOWN_RATIO=0.5 (compromis entre 0.4 et 0.6)
    - Embedding tying retry (exp11 crashed, jamais testé sur GPU)
    - Differential Transformer (complexe mais potentiellement high-impact)
  - Si exp14 MLP 3x marche :
    - Tester MLP 2.5x ou 2x pour encore plus de speed
    - Combiner MLP 3x avec DEPTH=9 (redistribuer les params économisés en plus de layers)
    - Tester GeGLU/SwiGLU à 3x ratio (SwiGLU a échoué à 4x, peut-être que 3x change la donne)
  - Note: la plupart des low-hanging fruits sont épuisés. Les gains > 0.003 vont être difficiles.
  - Note: Muon NS iterations (ns_steps=5) coûtent très peu en compute (matmuls batched 512×512 = ~0.05ms). Réduire ns_steps ne vaut pas le coup.
  - **Exp14 MLP 3x = 1.079602** (BEST EVER, -0.008 from baseline). Massive win. "More steps" strategy confirmed dominant.
  - **Exp15 teste MLP 2.5x** (hidden 1280 au lieu de 1536). Continue la stratégie gagnante.
  - Si exp15 MLP 2.5x marche :
    - Tester MLP 2x (1024 hidden) pour pousser encore plus loin
    - Combiner MLP 2.5x avec `torch.compile(mode="max-autotune")` pour encore plus de vitesse
  - Si exp15 MLP 2.5x échoue (régression) :
    - 3x est le sweet spot. Tester d'autres optimisations compute-réductrices :
    - `torch.compile(mode="max-autotune")` — zero risk, pourrait gagner 5-10% throughput
    - Réduire DEPTH=7 avec MLP 3x (moins de layers mais plus rapides)
    - Réduire n_embd (ASPECT_RATIO=56 → dim=448, mais 448/128=3.5 heads... arrondi nécessaire)
    - DEVICE_BATCH_SIZE=48 ou 64 pour mieux utiliser le GPU (exp1 run3 a testé 64 mais c'était avec 4x MLP, peut-être différent avec 2.5x)
  - **Exp15 MLP 2.5x = 1.082175** (+0.0026 vs 3x). 3x confirmed as sweet spot.
  - **Exp16 teste torch.compile(mode="max-autotune")** — CUDA graphs + autotuned Triton kernels. Zero architectural risk.
  - Si exp16 max-autotune marche :
    - Combiner avec DEPTH=7 (fewer layers + faster execution)
    - Tester DEVICE_BATCH_SIZE=48 avec MLP 3x (meilleure utilisation GPU avec le modèle plus petit)
  - Si exp16 max-autotune échoue/neutral :
    - DEPTH=7 avec MLP 3x (supprimer 1 layer = ~12.5% less compute, mais perte de capacité)
    - DEPTH=9 avec MLP 3x et smaller dim (ASPECT_RATIO=56 → dim=504→512, 4 heads, 9 layers: more layers but similar total compute)
    - Embedding tying retry (never successfully tested — exp11 crashed)
    - Schedule-Free Adam (replace entire warmup/warmdown scheduler)
    - GeGLU à 3x (SwiGLU échoué à 4x, mais 3x = moins de compute, potentiellement différent)

