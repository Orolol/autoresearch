#!/usr/bin/env bash
# ralph.sh — Orchestrateur autonome d'expériences ML
# Lance Claude en boucle pour itérer sur train.py et améliorer val_bpb.
#
# Usage: ./ralph.sh
#        ./ralph.sh --max 20   # limiter à 20 expériences
#
# Appuyer sur Ctrl+C (ou Echap dans le terminal) pour tout arrêter proprement.

set -uo pipefail
# PAS de set -m : on garde les enfants dans le même process group
# pour que Ctrl+C (SIGINT) les atteigne directement

ROOT="$(cd "$(dirname "$0")" && pwd)"
EXP_ROOT="$ROOT/experiments"
RESULTS_FILE="$EXP_ROOT/results.tsv"
BEST_TRAIN="$ROOT/train_best.py"
BEST_SCORE_FILE="$EXP_ROOT/best_score"
PROMPT_FILE="$ROOT/prompt.md"
RUN_LOGS="$ROOT/run_logs.md"
STOPPING=0

CHILD_PID=0

cleanup() {
    [[ "$STOPPING" -eq 1 ]] && return
    STOPPING=1
    trap '' INT TERM HUP  # ignorer les signaux pendant le cleanup

    echo ""
    echo "========================================="
    echo "  Arrêt demandé — nettoyage en cours..."
    echo "========================================="

    # Tuer le processus enfant direct et tout son arbre
    if [[ "$CHILD_PID" -gt 0 ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
        # Tuer tout l'arbre de processus via pkill -P (enfants récursifs)
        pkill -TERM -P "$CHILD_PID" 2>/dev/null || true
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        sleep 1
        pkill -9 -P "$CHILD_PID" 2>/dev/null || true
        kill -9 "$CHILD_PID" 2>/dev/null || true
    fi

    # Filet de sécurité : tuer tout train.py et claude qui traînent
    pkill -9 -f "python.*train\.py" 2>/dev/null || true
    pkill -9 -f "claude.*dangerously" 2>/dev/null || true

    echo "Tout est arrêté. Au revoir !"
    exit 0
}

trap cleanup INT TERM HUP

# --- Args ---
MAX_EXPERIMENTS=0 # 0 = infini
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max) MAX_EXPERIMENTS="$2"; shift 2 ;;
        *) echo "Usage: $0 [--max N]"; exit 1 ;;
    esac
done

# --- Prérequis ---
if ! command -v claude &>/dev/null; then
    echo "ERROR: 'claude' CLI not found. Install Claude Code first."
    exit 1
fi

if ! command -v uv &>/dev/null; then
    echo "ERROR: 'uv' not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "ERROR: prompt.md not found. Create it first."
    exit 1
fi

# --- Initialisation ---
mkdir -p "$EXP_ROOT"

if [[ ! -f "$RESULTS_FILE" ]]; then
    printf "experiment\tval_bpb\tpeak_vram_mb\tstatus\tdescription\n" > "$RESULTS_FILE"
fi

if [[ ! -f "$BEST_TRAIN" ]]; then
    cp "$ROOT/train.py" "$BEST_TRAIN"
fi

# Initialiser run_logs.md si absent
if [[ ! -f "$RUN_LOGS" ]]; then
    cat > "$RUN_LOGS" <<'EOF'
# Run Logs — Autoresearch

Journal des expériences lancées par l'agent.

---

EOF
fi

# --- Fonctions utilitaires ---
get_next_num() {
    local last
    last=$(find "$EXP_ROOT" -maxdepth 1 -type d -name 'experiment_*' \
        | grep -oP '\d+$' | sort -n | tail -1 2>/dev/null || echo "-1")
    echo $((last + 1))
}

extract_metric() {
    local file="$1" key="$2"
    grep "^${key}:" "$file" 2>/dev/null | tail -1 | awk '{print $2}' || echo ""
}

log_result() {
    local num="$1" val_bpb="$2" peak_vram="$3" status="$4" desc="$5"
    local vram_gb
    vram_gb=$(uv run python -c "print(f'{float(\"${peak_vram:-0}\") / 1024:.1f}')" 2>/dev/null || echo "0.0")
    printf "%s\t%s\t%s\t%s\t%s\n" "$num" "$val_bpb" "$vram_gb" "$status" "$desc" >> "$RESULTS_FILE"
}

log_run_start() {
    local num="$1" best="$2"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    cat >> "$RUN_LOGS" <<EOF

## Experiment #$num — $timestamp

- **Best val_bpb avant** : $best
- **Status** : en cours...

EOF
}

log_run_end() {
    local num="$1" val_bpb="$2" status="$3" desc="$4"
    # Mettre à jour le status dans run_logs.md
    sed -i "/## Experiment #$num/,/^## /{s/- \*\*Status\*\* : en cours.../- **Status** : $status (val_bpb: $val_bpb)\n- **Description** : $desc/}" "$RUN_LOGS" 2>/dev/null || true
}

# --- Baseline (experiment_0) ---
run_baseline() {
    echo "========================================="
    echo "  Baseline run (experiment_0)"
    echo "========================================="

    local exp_dir="$EXP_ROOT/experiment_0"
    mkdir -p "$exp_dir"

    cp "$BEST_TRAIN" "$ROOT/train.py"
    cp "$ROOT/train.py" "$exp_dir/train.py"
    cp "$PROMPT_FILE" "$exp_dir/prompt.md"
    echo "baseline" > "$exp_dir/report.md"

    log_run_start "0" "N/A (baseline)"

    echo "Running: uv run train.py ..."
    if uv run "$ROOT/train.py" > "$exp_dir/run.log" 2>&1; then
        local val_bpb peak_vram
        val_bpb=$(extract_metric "$exp_dir/run.log" "val_bpb")
        peak_vram=$(extract_metric "$exp_dir/run.log" "peak_vram_mb")

        if [[ -n "$val_bpb" ]]; then
            echo "$val_bpb" > "$BEST_SCORE_FILE"
            log_result "0" "$val_bpb" "$peak_vram" "keep" "baseline"
            log_run_end "0" "$val_bpb" "keep" "baseline"
            echo "Baseline: val_bpb=$val_bpb"
            return 0
        fi
    fi

    echo "ERROR: Baseline run failed. Check $exp_dir/run.log"
    tail -20 "$exp_dir/run.log" 2>/dev/null || true
    exit 1
}

if [[ ! -f "$BEST_SCORE_FILE" ]]; then
    run_baseline
fi

# --- Boucle principale ---
while true; do
    NUM=$(get_next_num)

    if [[ "$MAX_EXPERIMENTS" -gt 0 && "$NUM" -gt "$MAX_EXPERIMENTS" ]]; then
        echo "Max experiments ($MAX_EXPERIMENTS) reached. Stopping."
        break
    fi

    EXP_DIR="$EXP_ROOT/experiment_$NUM"
    mkdir -p "$EXP_DIR"

    BEST_SCORE=$(cat "$BEST_SCORE_FILE")

    # Préparer le working copy
    cp "$BEST_TRAIN" "$ROOT/train.py"
    cp "$BEST_TRAIN" "$EXP_DIR/train_before.py"
    cp "$PROMPT_FILE" "$EXP_DIR/prompt.md"

    # Construire l'historique récent (results.tsv + 5 derniers rapports)
    HISTORY="$(cat "$RESULTS_FILE")"
    RECENT_REPORTS=""
    for report in $(find "$EXP_ROOT" -name "report.md" -path "*/experiment_*/report.md" | sort -V | tail -5); do
        exp_name=$(basename "$(dirname "$report")")
        RECENT_REPORTS+="
### $exp_name
$(cat "$report")
"
    done

    echo ""
    echo "========================================="
    echo "  Experiment #$NUM"
    echo "  Best val_bpb: $BEST_SCORE"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================="

    # Logger le début du run
    log_run_start "$NUM" "$BEST_SCORE"

    # Lire le run_logs.md pour le passer à Claude
    RUN_LOGS_CONTENT=""
    if [[ -f "$RUN_LOGS" ]]; then
        RUN_LOGS_CONTENT="$(cat "$RUN_LOGS")"
    fi

    # --- Phase 1 : Claude modifie train.py (PAS de run d'entraînement) ---
    CLAUDE_PROMPT="$(cat <<PROMPT_EOF
Tu es un chercheur autonome en deep learning.

## Tes instructions
$(cat "$PROMPT_FILE")

## Contexte
- Expérience #$NUM
- Meilleur val_bpb actuel : $BEST_SCORE (plus bas = meilleur)
- Répertoire de travail : $ROOT
- Dossier de cette expérience : $EXP_DIR

## Historique des résultats
$HISTORY

## Rapports récents
$RECENT_REPORTS

## Journal des runs (run_logs.md)
$RUN_LOGS_CONTENT

## Règles strictes
- Tu ne modifies QUE train.py (prepare.py est read-only)
- Pas de nouvelles dépendances (uniquement ce qui est dans pyproject.toml)
- Tu ne lances PAS l'entraînement toi-même — c'est ralph.sh qui s'en charge après toi
- NE PAS exécuter uv run train.py. NE PAS lancer de run.

## À faire
1. Lis train.py et prepare.py pour comprendre le code actuel
2. Analyse l'historique et le journal des runs pour éviter de répéter des échecs
3. Choisis UNE idée d'amélioration (PRIORITÉ : changements architecturaux > optimisation code > hyperparamètres)
4. Écris ce que tu vas essayer dans $ROOT/run_logs.md (append, ne pas écraser). Inclus :
   - L'idée en une phrase
   - Pourquoi tu penses que ça va marcher
   - Les risques (crash, OOM, régression)
5. Modifie train.py avec ton changement
6. Vérifie que le code est syntaxiquement correct (python3 -c "import ast; ast.parse(open('$ROOT/train.py').read())")
7. Écris un rapport AVANT run dans $EXP_DIR/report.md :
   - Ligne 1 : description courte (UNE phrase, sans markdown heading)
   - Ce que tu as changé et pourquoi
8. Si tu as des notes ou conseils pour les prochaines itérations, modifie $ROOT/prompt.md

IMPORTANT : ton SEUL travail est de modifier train.py intelligemment. L'entraînement sera lancé automatiquement après.
Commence maintenant. Ne demande pas confirmation.
PROMPT_EOF
    )"

    CLAUDE_LOG="$EXP_DIR/claude_output.log"
    claude -p "$CLAUDE_PROMPT" --dangerously-skip-permissions > "$CLAUDE_LOG" 2>&1 &
    CHILD_PID=$!
    wait "$CHILD_PID" 2>/dev/null
    CHILD_PID=0
    [[ "$STOPPING" -eq 1 ]] && exit 0
    tail -5 "$CLAUDE_LOG" 2>/dev/null || true

    # Vérifier que train.py est syntaxiquement valide avant de lancer
    if ! uv run python -c "import ast; ast.parse(open('$ROOT/train.py').read())" 2>/dev/null; then
        echo ">>> SYNTAX ERROR in train.py — skipping run"
        cp "$BEST_TRAIN" "$ROOT/train.py"
        DESC="syntax error"
        if [[ -f "$EXP_DIR/report.md" ]]; then
            DESC=$(head -1 "$EXP_DIR/report.md" | sed 's/^#* *//' | tr '\t' ' ' | cut -c1-120)
        fi
        log_result "$NUM" "0.000000" "0" "crash" "$DESC (syntax error)"
        log_run_end "$NUM" "0.000000" "crash" "$DESC (syntax error)"
        echo "========================================="
        echo ""
        continue
    fi

    # Sauvegarder la version de train.py avant le run
    cp "$ROOT/train.py" "$EXP_DIR/train.py"

    # --- Phase 2 : ralph lance l'entraînement ---
    echo "  Running training..."
    RUN_LOG="$EXP_DIR/run.log"
    timeout 600 uv run "$ROOT/train.py" > "$RUN_LOG" 2>&1 &
    CHILD_PID=$!
    wait "$CHILD_PID" 2>/dev/null
    CHILD_PID=0
    [[ "$STOPPING" -eq 1 ]] && exit 0

    VAL_BPB=$(extract_metric "$RUN_LOG" "val_bpb")
    PEAK_VRAM=$(extract_metric "$RUN_LOG" "peak_vram_mb")

    DESC="experiment $NUM"
    if [[ -f "$EXP_DIR/report.md" ]]; then
        DESC=$(head -1 "$EXP_DIR/report.md" | sed 's/^#* *//' | tr '\t' ' ' | cut -c1-120)
    fi

    if [[ -n "$VAL_BPB" && "$VAL_BPB" != "0" ]]; then
        IMPROVED=$(uv run python -c "print('yes' if float('$VAL_BPB') < float('$BEST_SCORE') else 'no')")

        # Mettre à jour le rapport avec les résultats
        if [[ -f "$EXP_DIR/report.md" ]]; then
            printf "\n\n## Résultats\n- val_bpb: %s\n- peak_vram_mb: %s\n- status: %s\n" \
                "$VAL_BPB" "$PEAK_VRAM" "$( [[ "$IMPROVED" == "yes" ]] && echo "IMPROVED" || echo "discarded" )" \
                >> "$EXP_DIR/report.md"
        fi

        if [[ "$IMPROVED" == "yes" ]]; then
            cp "$ROOT/train.py" "$BEST_TRAIN"
            echo "$VAL_BPB" > "$BEST_SCORE_FILE"
            log_result "$NUM" "$VAL_BPB" "$PEAK_VRAM" "keep" "$DESC"
            log_run_end "$NUM" "$VAL_BPB" "keep" "$DESC"
            echo ">>> IMPROVED: $BEST_SCORE -> $VAL_BPB <<<"
        else
            cp "$BEST_TRAIN" "$ROOT/train.py"
            log_result "$NUM" "$VAL_BPB" "$PEAK_VRAM" "discard" "$DESC"
            log_run_end "$NUM" "$VAL_BPB" "discard" "$DESC"
            echo ">>> No improvement: $VAL_BPB (best: $BEST_SCORE)"
        fi
    else
        cp "$BEST_TRAIN" "$ROOT/train.py"
        # Extraire les dernières lignes d'erreur
        echo "--- Last 10 lines of run.log ---"
        tail -10 "$RUN_LOG" 2>/dev/null || true
        echo "---"
        log_result "$NUM" "0.000000" "0" "crash" "$DESC"
        log_run_end "$NUM" "0.000000" "crash" "$DESC"
        echo ">>> CRASH: no val_bpb found"
    fi

    echo "========================================="
    echo ""
done
