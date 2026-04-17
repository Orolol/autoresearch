# Autoresearch — Instructions pour l'agent

Tu es un chercheur autonome qui optimise un modèle de langage (GPT) pour obtenir le meilleur val_bpb possible en 5 minutes de training sur un seul GPU. ultrathink

## Hardware

- **GPU** : NVIDIA RTX 5090 (architecture Blackwell), ~32 GB VRAM
- **Précision** : BF16 par défaut, FP8 disponible (Blackwell supporte nativement FP8)
- **CUDA** : 13.1, PyTorch 2.10+
- `torch.compile()` est activé — les custom kernels Triton sont supportés
- **SDPA path** (non-Hopper) : pas de sliding window. Toutes les layers font full causal attention. Le WINDOW_PATTERN n'a AUCUN effet sur le compute.

## Priority queue — seules pistes restantes

| # | Piste | Pourquoi | Next step |
|---|-------|----------|-----------|
| 1 | **WD=0.3** | WD=0.2 jamais AB-testé vs 0.3. Courbe: WD=0.0 (+0.004), WD=0.1 (+0.002), WD=0.2 (best). MLP ramp axis dead (exp198 +0.002, 5 total experiments). Chunk attention axis EXHAUSTED (8 consecutive failures post-exp193). | Si pire: WD=0.2 confirmé optimal, axe mort. Si mieux: tester WD=0.4. |
| 2 | **Pathway-specific cross-chunk gating** (exp199 testing) | Separate learned gate for cross-chunk recurrent contribution in local layers. `(intra + cross_gate * cross) * attn_gate` instead of `(intra + cross) * attn_gate`. Per-position per-head control over local vs global pathway. 128 new params. | If improves: axis validated, try extending to global layers (attn bias gating). If neutral: existing attn_gate/attn_temp provide sufficient control, cross-chunk combination axis dead. |
| 3 | **Partitioned Hyperconnections** | modded-nanogpt PR#230 (Feb 2026, -45 steps). 2-lane residual: attn lit lane0, MLP lit lane1. exp189 tried: +0.007 regression. Maybe wrong implementation? | Complexe à implémenter proprement. exp189 failed, but may be worth retrying with different init/structure. |

**Note (exp167)** : Learnable softcap — 1.063240 (neutre). Softcap axis épuisé (3 discrete + 1 continuous test). Softcap=13 est le vrai optimum.

**Note (exp171-172)** : LR Floor — FINAL_LR_FRAC=0.1 est optimal (exp171: 1.062310, new best). FINAL_LR_FRAC=0.15 (exp172: 1.064116, pire). Axe LR floor épuisé.

**Note (exp177)** : Low-rank cross-block recurrent state — 1.066348 (+0.004). Full-rank exp152 (+0.006) also failed. Cross-block addons for local layers are DEAD (softmax + linear recurrence doesn't help, regardless of rank/decay/gating).

**Note (exp178)** : Sigmoid attention for local layers — 1.082753 (+0.020). Replacing softmax with sigmoid is very destructive. Linear attention (exp179) is different (has decay, recurrent state, QK-norm) but risk is real.

**Note FP8** : ABANDONNÉ après 4+ crashes (exp121, exp169, exp170x3). torch._scaled_mm semble cassé sur Blackwell/CUDA 13.1/PyTorch 2.10+. Les tentatives ont inclus : all linears, MLP-only, col-major fix, setup_context, @torch.compiler.disable. Ne plus retenter sauf si PyTorch met à jour le support FP8 Blackwell.

## Méta-leçons (167 expériences)

**Ce qui marche** : réduire le compute par step (MLP 3x, local attention), ajouter de l'information via gating sparse (VE bookend, attention gate, token shift), tuning fin d'un seul hyperparamètre à la fois.

**Ce qui ne marche PAS** :
- **Ajouter des paramètres/compute** : DEPTH=9, parallel attn+MLP, wider heads — toujours pire car réduit le nombre de steps.
- **Régularisation agressive** : label smoothing, EMA, remove x0_lambdas — le training est trop court pour overfitter.
- **Alternatives à Muon/ReluSquared/RoPE** : SwiGLU, GeGLU, DiffAttn, xPos, Schedule-Free Adam — tous pires à cette échelle.
- **Gating sur des états sans structure** : MLP gate, skip gate, V scaling — le gating ne marche que sur attention heads (structure sémantique).
- **Recurrence/cross-block** : cross-block linear attention, cross-layer MLP residual — token shift + full-attention bookends suffisent.

**Seuil de significativité** : variance ~0.003. Ne retenter que si delta > 0.003.

Après 167 exp, le modèle est probablement proche de son plafond (~1.062) à ~830 steps. Les gains futurs requièrent FP8 (plus de steps) ou une attention fondamentalement différente (DeltaNet/GLA).

## Philosophie
- Privilégie les changements audacieux sur les composants plutôt que le micro-tuning
- Une idée architecturale nouvelle vaut mieux que 10 tweaks d'hyperparamètres
- Les petites améliorations (~0.001 val_bpb) ne valent pas 20 lignes de complexité
- Si tu supprimes du code et que le score est égal ou meilleur, c'est une victoire
- N'hésite pas à tenter des choses radicales — on peut toujours revert
- Les crashes sont automatiquement analysés et potentiellement corrigés — ne laisse pas la peur du crash limiter ton ambition
- Après 3+ expériences consécutives de tuning d'hyperparamètres, FORCE-toi à tenter un changement architectural
- Regarde les crash_analysis.md des expériences précédentes pour comprendre ce qui a crashé et pourquoi
- Utilise internet pour aller piocher des idées, consulte des papiers de recherche
- Reddit et Arxiv sont des bonnes ressources

**Note (exp179)** : Chunkwise linear attention for local layers — 1.050748 (-0.012 from 1.062310). MASSIVE improvement. Biggest single-experiment gain in 179 experiments. Cross-chunk recurrent KV state with learnable per-head exponential decay. exp180: all 8 layers = +0.044 regression (global layers NEED softmax). exp181: GLA input-dependent decay = NaN (cumsum(log_gamma) numerically unstable — avoid log-space decay computations).

**Note (exp193)** : RetNet-style hybrid attention — 1.049078 (-0.0017 from 1.050748). Replaced all-linear intra-chunk with softmax causal attention + learned log-decay positional bias (ALiBi-style). Cross-chunk linear recurrence unchanged. Softmax gives sharper, more selective local attention. 6 consecutive pure-linear improvements failed (183, 189-192) before this paradigm shift worked. Post-exp193 failed attempts: exp194 chunk_size=512 (+0.003), exp195 memory tokens (+0.056), exp196 DeltaNet (+0.015), exp197 RoPE-free cross-chunk (+0.011), exp198 MLP ramp (+0.002). Chunk attention axis EXHAUSTED (8 failures). MLP distribution axis DEAD (5 experiments).

## Config actuelle (best = 1.049078, exp193)

DEPTH=8, ASPECT_RATIO=64, HEAD_DIM=128 (4 heads), non-uniform MLP (1.5x early, 4.5x late) ReluSquared, softcap=13, z-loss=1e-4, TOTAL_BATCH_SIZE=2^18, DEVICE_BATCH_SIZE=32, EMBEDDING_LR=0.6, MATRIX_LR=0.04, WEIGHT_DECAY=0.2, WARMDOWN_RATIO=0.7, multi-scale RoPE (2500,10000,40000,160000), U-Net skip connections, VE bookend (first+last layer, shared weights, separate gates, full n_embd width), VE WD=0.002, x0_lambdas, RWKV-style token shift (K+V, per-channel sigmoid gates), sparse attention gate (gate_dim=2*n_head=8, per-head output gating), hybrid local attention (chunk_size=256, softmax intra-chunk + log-decay bias + linear cross-chunk recurrence, middle layers 2-5 local, bookend layers 0,1,6,7 full softmax SDPA).

## Leçons confirmées (95 expériences)

### Ce qui a marché (gardé dans le baseline)
- **MLP 3x** (exp14, -0.008) : réduction compute -> plus de steps. Le gain le plus important.
- **Softcap 13** (exp45, -0.001) : meilleur gradient flow à travers tanh.
- **Z-loss 1e-4** (exp9) : stabilise les logits. 1e-4 est l'optimum exact (5e-5 pire, 2e-4 pire).
- **U-Net skip connections** (exp71, -0.001) : skip_lambdas init=0.0, learned via AdamW.
- **Multi-scale RoPE** (exp80, -0.0002) : per-head bases (2500,10000,40000,160000).
- **VE bookend** (exp84, -0.004) : VE sur first+last layer, shared weights, separate gates.

### DEPTH=9 — confirmé pire (exp97 = 1.079870, +0.007)
8 tentatives. Exp86-88 = bug IndexError. Exp92-95 = infra RunPod crash. Exp97 = première run réussie : 1.079870 (+0.007 vs best 1.072821). La réduction de steps (~750 vs ~830) l'emporte sur le gain de capacité. DEPTH=9 est mort.

## Idées mortes — NE PAS retenter

**Architecture** : parallel attn+MLP (+0.034), DEPTH=7 (+0.033), DEPTH=9 (exp97 +0.007), HEAD_DIM=64 (+0.012), factored lm_head (+0.016), MLP 2.5x (+0.003), pre-output MLP (+0.003), mid-MLP norm (+0.002), qk_scales (+0.002), VE all layers (+0.009), Key Embedding (exp113 neutral), per-head output scales (exp111 +0.001), per-dimension lambdas (exp114 +0.009), WTE-as-value identity (exp118 +0.013), dense attn output gating (exp131 neutral — sparse variant worked), DiffAttn V-halving (exp105 +0.009), sandwich norm (exp106 +0.008), causal depthwise conv+MLP (exp107 +0.023), Soft MoD (exp116 +0.012), affine RMSNorm (exp132 +0.003), partial RoPE (exp157 +0.0025), cross-layer MLP residual (exp158 NaN), hybrid local/global heads (exp159 +0.0024), learned absolute pos emb (exp160 +0.004), layer-conditional activation (exp161 +0.004), value centroid injection (exp162 +0.008), adaptive x0 injection (exp163 +0.016)

**Attention/gating** : sparse MLP gate (exp142 +0.004), sparse skip gating (exp148 +0.007), dynamic key temperature (exp154 +0.005), per-head V scaling (exp155 +0.006), cross-block linear recurrence (exp152 +0.006)

**Activations** : SwiGLU (+0.004), GeluSquared (+0.006). ReluSquared est le meilleur à cette échelle.

**Optimizer** : Muon momentum warmup 300->100 (+0.0007), Muon beta2=0.99 (+0.001), momentum ceiling 0.97 (+0.004), lm_head Muon (crash), MATRIX_LR=0.06 (neutral), gradient clipping (+0.010), decouple AdamW/Muon LR (+0.013), Schedule-Free AdamW (exp110 +0.017), UNEMBEDDING_LR=0.01 (exp125 +0.003), EMBEDDING_LR=0.4 (exp134 neutral), EMBEDDING_LR=0.8 (exp166 neutral). **Muon est parfaitement tuné.**

**Training** : EMA (+0.016), label smoothing (+0.040), cosine warmdown (+0.004), WARMUP_RATIO=0.01 (+0.004), WEIGHT_DECAY=0.0 (+0.004), DEVICE_BATCH_SIZE=64 (+0.016), TOTAL_BATCH_SIZE=2^17 (+0.011), torch.compile modes (crash/pire), embedding tying (crash), constant WD (exp139 +0.001), focal loss (exp140 +0.002). reduce-overhead compile (exp164) en test.

**Régularisation** : softcap=30 (+0.005), softcap=11 (+0.001), z-loss 5e-5 (+0.002), z-loss 2e-4 (+0.001), remove x0_lambdas (+0.020), remove VE (+0.023), VE WD=0.005 (exp119 neutral), VE betas 0.9 (exp120 neutral), wte WD (exp117 neutral), x0_lambdas init=0.05 (exp165 neutral)

**Token shift** : Q shift (exp127 +0.008), V-only shift (exp128 +0.007). K+V est l'optimum.

**Positional/batch** : RoPE base=1000 (+0.013), wider range 500x (exp109 neutral), wider range 200x (exp135 neutral). Multi-scale RoPE 64x est l'optimum. TOTAL_BATCH_SIZE=2^18 est l'optimum exact.

## Bug fix documenté

U-Net skip pour odd depths : `if i >= half:` -> `if i >= half and mirror < half:`. Sans ce fix, DEPTH=9 crash avec IndexError (mirror=4, skip_lambdas size=4, index out of range).

## Note sur MoE
torch.compile(dynamic=False) est incompatible avec sparse MoE routing (token counts dynamiques). Le seul MoE compile-friendly calcule TOUS les experts pour TOUS les tokens = double le compute MLP. MoE est effectivement mort sauf trick compile-friendly.
