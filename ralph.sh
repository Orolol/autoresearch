#!/usr/bin/env bash
# ralph.sh — Multi-project autoresearch orchestrator.
#
# Subcommands:
#   ./ralph.sh list                       # list projects and their current best score
#   ./ralph.sh new <name>                 # scaffold projects/<name>/ from projects/_template/
#   ./ralph.sh run <project> [--max N]    # run the RALPH loop for <project>
#
# Ctrl+C stops cleanly (kills the current child process tree).

set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECTS_ROOT="$ROOT/projects"
STOPPING=0
MAX_FIX_ATTEMPTS=1              # nb max de fix par crash analysis

CHILD_PID=0

cleanup() {
    [[ "$STOPPING" -eq 1 ]] && return
    STOPPING=1
    trap '' INT TERM HUP

    echo ""
    echo "========================================="
    echo "  Arrêt demandé — nettoyage en cours..."
    echo "========================================="

    if [[ "$CHILD_PID" -gt 0 ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
        pkill -TERM -P "$CHILD_PID" 2>/dev/null || true
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        sleep 1
        pkill -9 -P "$CHILD_PID" 2>/dev/null || true
        kill -9 "$CHILD_PID" 2>/dev/null || true
    fi

    pkill -9 -f "python.*train\.py" 2>/dev/null || true
    pkill -9 -f "claude.*dangerously" 2>/dev/null || true

    echo "Tout est arrêté. Au revoir !"
    exit 0
}
trap cleanup INT TERM HUP

# --- Args ---
MAX_EXPERIMENTS=0 # 0 = infini
USE_REMOTE=0
REMOTE_GPU="RTX_5090"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max) MAX_EXPERIMENTS="$2"; shift 2 ;;
        --remote) USE_REMOTE=1; shift ;;
        --gpu) REMOTE_GPU="$2"; shift 2 ;;
        *) echo "Usage: $0 [--max N] [--remote] [--gpu TYPE]"; exit 1 ;;
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

if [[ "$USE_REMOTE" -eq 1 ]] && [[ -z "${RUNPOD_API_KEY:-}" ]]; then
    echo "ERROR: RUNPOD_API_KEY is required when using --remote."
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

# --- Training runner (local or remote) ---
run_training() {
    local log_file="$1"
    if [[ "$USE_REMOTE" -eq 1 ]]; then
        RUNPOD_GPU="$REMOTE_GPU" timeout 900 uv run "$ROOT/train_remote.py" > "$log_file" 2>&1
    else
        timeout 600 uv run "$ROOT/train.py" > "$log_file" 2>&1
    fi
}

# --- Fonctions utilitaires ---
get_next_num() {
    local last
    last=$(find "$EXP_ROOT" -maxdepth 1 -type d -name 'experiment_*' \
        | grep -oP '\d+$' | sort -n | tail -1 2>/dev/null || echo "-1")
    echo $((last + 1))
}

# read_toml_list <file> <python-expr-returning-iterable>
read_toml_list() {
    local file="$1" expr="$2"
    uv run python - "$file" <<PY
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib
with open(sys.argv[1], "rb") as f:
    data = tomllib.load(f)
for x in $expr:
    print(x)
PY
}

# --- Subcommands ---

cmd_list() {
    [[ $# -gt 0 ]] && { echo "Usage: $0 list"; exit 1; }

    if [[ ! -d "$PROJECTS_ROOT" ]]; then
        echo "No projects/ directory. Use '$0 new <name>' to create one."
        return
    fi

    printf "%-24s  %-16s  %-10s  %-14s  %s\n" "name" "metric" "direction" "best" "#experiments"
    printf "%-24s  %-16s  %-10s  %-14s  %s\n" "----" "------" "---------" "----" "------------"

    local any=0
    for toml in "$PROJECTS_ROOT"/*/project.toml; do
        [[ -e "$toml" ]] || continue
        local pdir name
        pdir="$(dirname "$toml")"
        name="$(basename "$pdir")"
        [[ "$name" == "_template" ]] && continue
        any=1

<<<<<<< HEAD
log_run_end() {
    local num="$1" val_bpb="$2" status="$3" desc="$4"
    # Mettre à jour le status dans run_logs.md
    sed -i "/## Experiment #$num/,/^## /{s/- \*\*Status\*\* : en cours.../- **Status** : $status (val_bpb: $val_bpb)\n- **Description** : $desc/}" "$RUN_LOGS" 2>/dev/null || true
}

# --- Crash analysis : lance Claude pour diagnostiquer un crash ---
run_crash_analysis() {
    local exp_dir="$1"
    local analysis_file="$exp_dir/crash_analysis.md"
    local analysis_log="$exp_dir/crash_analysis_claude.log"

    local CRASH_PROMPT
    CRASH_PROMPT="$(cat <<CRASH_EOF
Tu es un expert en diagnostic de crashes ML. Un script d'entraînement a crashé. Ton travail : classifier le crash et éventuellement corriger le code.

## Fichiers à lire
- $exp_dir/run.log (le log d'entraînement — concentre-toi sur les 200 DERNIÈRES lignes pour l'erreur)
- $exp_dir/train.py (le code qui a été exécuté)
- $exp_dir/train_before.py (le code AVANT les changements de cette expérience)
- $exp_dir/report.md (ce que l'expérience essayait de faire)

## Ta tâche
1. Lis le run.log (200 dernières lignes) et identifie l'erreur
2. Lis train.py et train_before.py pour comprendre ce qui a changé
3. Classifie le crash :
   - INFRA : problème d'infrastructure transitoire (RunPod init, CUDA driver init, timeout réseau, worker qui ne démarre pas, timeout de queue). Le code est OK, il faut juste retenter.
   - FIX : bug de code que tu peux corriger (mauvaise shape de tenseur, import manquant, mauvais indexing, incompatibilité torch.compile, etc.). Tu DOIS corriger train.py si tu choisis FIX.
   - SKIP : problème fondamental non fixable simplement (l'approche elle-même est cassée, dépendance manquante, conflit profond avec torch.compile).

## Format de sortie
Écris ton analyse dans $analysis_file avec ce format EXACT :

Ligne 1 : CLASSIFICATION: <INFRA|FIX|SKIP>
Ligne 2 : vide
Ligne 3+ : Explication de ce qui s'est passé et pourquoi tu as choisi cette classification.

## Règles
- Si CLASSIFICATION est FIX : tu DOIS aussi modifier $ROOT/train.py avec le fix. Ne corrige QUE le bug, ne change pas l'idée expérimentale.
- Si CLASSIFICATION est INFRA ou SKIP : NE modifie AUCUN fichier autre que $analysis_file.
- Sois concis. Concentre-toi sur la cause racine.
- Vérifie tout fix avec : python3 -c "import ast; ast.parse(open('$ROOT/train.py').read())"

Commence maintenant. Ne demande pas confirmation.
CRASH_EOF
    )"

    printf '%s' "$CRASH_PROMPT" > "$exp_dir/crash_analysis_prompt.txt"
    claude -p "$CRASH_PROMPT" --dangerously-skip-permissions > "$analysis_log" 2>&1 &
    CHILD_PID=$!
    wait "$CHILD_PID" 2>/dev/null
    CHILD_PID=0
    [[ "$STOPPING" -eq 1 ]] && exit 0

    # Parser la classification depuis crash_analysis.md
    if [[ -f "$analysis_file" ]]; then
        head -1 "$analysis_file" | grep -oP '(?<=CLASSIFICATION: )\w+' || echo "UNKNOWN"
    else
        echo "UNKNOWN"
    fi
}

# --- Évaluation des résultats (facteur commun) ---
evaluate_result() {
    local num="$1" val_bpb="$2" peak_vram="$3" best_score="$4" desc="$5" suffix="${6:-}"
    local improved
    improved=$(uv run python -c "print('yes' if float('$val_bpb') < float('$best_score') else 'no')")

    if [[ -f "$EXP_DIR/report.md" ]]; then
        printf "\n\n## Résultats%s\n- val_bpb: %s\n- peak_vram_mb: %s\n- status: %s\n" \
            "$suffix" "$val_bpb" "$peak_vram" "$( [[ "$improved" == "yes" ]] && echo "IMPROVED" || echo "discarded" )" \
            >> "$EXP_DIR/report.md"
    fi

    if [[ "$improved" == "yes" ]]; then
        cp "$ROOT/train.py" "$BEST_TRAIN"
        echo "$val_bpb" > "$BEST_SCORE_FILE"
        log_result "$num" "$val_bpb" "$peak_vram" "keep" "$desc"
        log_run_end "$num" "$val_bpb" "keep" "$desc"
        echo ">>> IMPROVED$suffix: $best_score -> $val_bpb <<<"
    else
        cp "$BEST_TRAIN" "$ROOT/train.py"
        log_result "$num" "$val_bpb" "$peak_vram" "discard" "$desc"
        log_run_end "$num" "$val_bpb" "discard" "$desc"
        echo ">>> No improvement$suffix: $val_bpb (best: $best_score)"
    fi
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

    echo "Running: training..."
    run_training "$exp_dir/run.log" &
    CHILD_PID=$!
    wait "$CHILD_PID" 2>/dev/null
    local rc=$?
    CHILD_PID=0
    [[ "$STOPPING" -eq 1 ]] && exit 0

    if [[ "$rc" -eq 0 ]]; then
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

    echo ""
    echo "========================================="
    echo "  Experiment #$NUM"
    echo "  Best val_bpb: $BEST_SCORE"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================="

    # Logger le début du run
    log_run_start "$NUM" "$BEST_SCORE"

    # --- Phase 1 : Claude modifie train.py (PAS de run d'entraînement) ---
    # Prompt court — Claude lit les fichiers lui-même pour éviter "Argument list too long"
    CLAUDE_PROMPT="$(cat <<PROMPT_EOF
Tu es un chercheur autonome en deep learning. Expérience #$NUM.

## Étape 1 : Lis ces fichiers
- $PROMPT_FILE (tes instructions complètes)
- $ROOT/train.py (le code à modifier)
- $ROOT/prepare.py (read-only, pour comprendre les données)
- $RESULTS_FILE (historique des résultats)
- $RUN_LOGS (journal des runs précédents — lis les 50 dernières lignes)
- Les 5 derniers rapports dans $EXP_ROOT/experiment_*/report.md
- Les crash analyses récentes dans $EXP_ROOT/experiment_*/crash_analysis.md (s'il y en a — lis-les pour comprendre pourquoi des expériences ont crashé)

## Contexte
- Meilleur val_bpb actuel : $BEST_SCORE (plus bas = meilleur)
- Répertoire de travail : $ROOT
- Dossier de cette expérience : $EXP_DIR

## Philosophie d'expérimentation
- Sois AUDACIEUX. Les micro-optimisations (+0.001) ne valent pas le coût d'une itération entière.
- Préfère les changements architecturaux radicaux aux tweaks d'hyperparamètres.
- Les crashes sont OK — ils sont automatiquement diagnostiqués et potentiellement corrigés par un système d'analyse post-crash.
- Si les 3 dernières expériences étaient des tweaks d'hyperparamètres, tu DOIS tenter quelque chose de structurellement différent.
- Ne répète JAMAIS une idée morte listée dans prompt.md, même avec une "petite variation".
- Cherche de l'inspiration dans la littérature récente (arxiv, modded-nanogpt, etc.).
- Un crash audacieux vaut mieux qu'un +0.001 timide.
=======
        local metric direction best count
        metric="$(read_toml_str "$toml" "data['metric']['key']")"
        direction="$(read_toml_str "$toml" "data['metric']['direction']")"
        if [[ -f "$pdir/experiments/best_score" ]]; then
            best="$(cat "$pdir/experiments/best_score")"
        else
            best="—"
        fi
        count=$(find "$pdir/experiments" -maxdepth 1 -type d -name 'experiment_*' 2>/dev/null | wc -l)
        printf "%-24s  %-16s  %-10s  %-14s  %s\n" "$name" "$metric" "$direction" "$best" "$count"
    done

    if [[ "$any" -eq 0 ]]; then
        echo "(no projects yet; only _template exists)"
    fi
    return 0
}

cmd_new() {
    [[ $# -ne 1 ]] && { echo "Usage: $0 new <name>"; exit 1; }
    local name="$1"

    if ! [[ "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
        echo "ERROR: project name must match [a-zA-Z0-9][a-zA-Z0-9_-]* — got '$name'."
        exit 1
    fi
    if [[ "$name" == "_template" ]]; then
        echo "ERROR: '_template' is a reserved name."
        exit 1
    fi

    local target="$PROJECTS_ROOT/$name"
    local template="$PROJECTS_ROOT/_template"

    [[ -e "$target" ]]     && { echo "ERROR: $target already exists."; exit 1; }
    [[ ! -d "$template" ]] && { echo "ERROR: $template not found."; exit 1; }

    cp -r "$template" "$target"
    # Replace only the name = "_template" line in the new project.toml.
    sed -i "s/^name = \"_template\"$/name = \"$name\"/" "$target/project.toml"

    cat <<EOF
Scaffolded: $target
>>>>>>> 6889b79 (feat(ralph): rewrite as subcommand dispatcher with list + new)

Next steps:
  1. Edit $target/project.toml (metric.key, direction, extra_keys)
  2. Implement $target/prepare.py (data + tokenizer)
  3. Implement $target/train.py (training loop; print <metric.key>: <float>)
  4. Fill $target/prompt.md (hardware, exploration priorities, philosophy)

<<<<<<< HEAD
## À faire
1. Lis les fichiers ci-dessus pour comprendre le code et l'historique
2. Analyse l'historique et le journal des runs pour éviter de répéter des échecs
3. Choisis UNE idée d'amélioration AMBITIEUSE (PRIORITÉ : innovations architecturales radicales > changements structurels > hyperparamètres). Si l'historique montre 3+ tweaks récents, tu DOIS tenter un changement architectural.
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
=======
When ready: $0 run $name
EOF
}
>>>>>>> 6889b79 (feat(ralph): rewrite as subcommand dispatcher with list + new)

cmd_run() {
    local project="" max=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --max) max="$2"; shift 2 ;;
            -h|--help) echo "Usage: $0 run <project> [--max N]"; exit 0 ;;
            -*) echo "Unknown flag: $1"; exit 1 ;;
            *) [[ -n "$project" ]] && { echo "Unexpected arg: $1"; exit 1; }
               project="$1"; shift ;;
        esac
    done
    [[ -z "$project" ]] && { echo "Usage: $0 run <project> [--max N]"; exit 1; }

    local pdir="$PROJECTS_ROOT/$project"
    local toml="$pdir/project.toml"
    [[ -f "$toml" ]] || { echo "ERROR: $toml not found."; exit 1; }

    require_cmd claude

    # Read project config.
    local metric_key direction train_cmd timeout_s
    metric_key="$(read_toml_str "$toml" "data['metric']['key']")"
    direction="$(read_toml_str "$toml" "data['metric']['direction']")"
    train_cmd="$(read_toml_str "$toml" "data['run']['train_cmd']")"
    timeout_s="$(read_toml_str "$toml" "data['run']['timeout_s']")"
    case "$direction" in minimize|maximize) ;; *)
        echo "ERROR: invalid metric.direction='$direction' in $toml."; exit 1 ;;
    esac

    # extra_keys as a bash array.
    local extra_keys=()
    while IFS= read -r k; do
        [[ -n "$k" ]] && extra_keys+=("$k")
    done < <(read_toml_list "$toml" "data['metric'].get('extra_keys', [])")

    local exp_root="$pdir/experiments"
    local results_file="$exp_root/results.tsv"
    local best_train="$pdir/train_best.py"
    local best_score_file="$exp_root/best_score"
    local prompt_file="$pdir/prompt.md"
    local run_logs="$pdir/run_logs.md"
    local train_py="$pdir/train.py"

    [[ -f "$prompt_file" ]] || { echo "ERROR: $prompt_file not found."; exit 1; }
    [[ -f "$train_py" ]]    || { echo "ERROR: $train_py not found."; exit 1; }

    mkdir -p "$exp_root"

    # Initialize results.tsv header (columns: experiment, <metric.key>, <extra_keys...>, status, description).
    if [[ ! -f "$results_file" ]]; then
        { printf "experiment\t%s" "$metric_key"
          for k in "${extra_keys[@]}"; do printf "\t%s" "$k"; done
          printf "\tstatus\tdescription\n"
        } > "$results_file"
    fi

    [[ -f "$best_train" ]] || cp "$train_py" "$best_train"

    if [[ ! -f "$run_logs" ]]; then
        cat > "$run_logs" <<EOF
# Run Logs — $project

Journal des expériences lancées par l'agent.

---
EOF
    fi

    # --- utility: extract "^KEY:" from a log file (last match) ---
    extract_metric() {
        local file="$1" key="$2"
        grep "^${key}:" "$file" 2>/dev/null | tail -1 | awk '{print $2}'
    }

    # --- utility: return "yes" if new beats best per direction ---
    is_improved() {
        local new="$1" best="$2"
        uv run python - "$new" "$best" "$direction" <<'PY'
import sys
new, best, direction = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    n, b = float(new), float(best)
except ValueError:
    print("no"); sys.exit()
if direction == "minimize":
    print("yes" if n < b else "no")
else:
    print("yes" if n > b else "no")
PY
    }

    # --- utility: append a row to results.tsv ---
    log_result() {
        # args: num score status desc extra_1 extra_2 ...
        local num="$1" score="$2" status="$3" desc="$4"; shift 4
        { printf "%s\t%s" "$num" "$score"
          for v in "$@"; do printf "\t%s" "$v"; done
          printf "\t%s\t%s\n" "$status" "$desc"
        } >> "$results_file"
    }

    log_run_start() {
        local num="$1" best="$2"
        local ts; ts=$(date '+%Y-%m-%d %H:%M:%S')
        cat >> "$run_logs" <<EOF

## Experiment #$num — $ts

- **Best $metric_key avant** : $best
- **Status** : en cours...

EOF
    }

    log_run_end() {
        local num="$1" score="$2" status="$3" desc="$4"
        # Range ends at next "## " heading; for the newest experiment it extends to EOF,
        # which is safe because "- **Status** : en cours..." is unique per experiment block.
        sed -i "/## Experiment #$num/,/^## /{s|- \*\*Status\*\* : en cours\.\.\.|- **Status** : $status ($metric_key: $score)\n- **Description** : $desc|}" "$run_logs" 2>/dev/null || true
    }

    get_next_num() {
        local last
        last=$(find "$exp_root" -maxdepth 1 -type d -name 'experiment_*' \
            | grep -oP '\d+$' | sort -n | tail -1 2>/dev/null || echo "-1")
        echo $((last + 1))
    }

    # --- Baseline (experiment_0) if best_score absent ---
    if [[ ! -f "$best_score_file" ]]; then
        echo "========================================="
        echo "  [$project] Baseline run (experiment_0)"
        echo "========================================="
        local exp_dir="$exp_root/experiment_0"
        mkdir -p "$exp_dir"
        cp "$best_train" "$train_py"
        cp "$train_py" "$exp_dir/train.py"
        cp "$prompt_file" "$exp_dir/prompt.md"
        echo "baseline" > "$exp_dir/report.md"
        log_run_start "0" "N/A (baseline)"
        echo "Running: $train_cmd (cwd=$pdir, timeout=${timeout_s}s) ..."
        ( cd "$pdir" && timeout "$timeout_s" bash -c "$train_cmd" ) > "$exp_dir/run.log" 2>&1 &
        CHILD_PID=$!
        wait "$CHILD_PID" 2>/dev/null
        CHILD_PID=0
        [[ "$STOPPING" -eq 1 ]] && exit 0

        local score; score="$(extract_metric "$exp_dir/run.log" "$metric_key")"
        if [[ -z "$score" ]]; then
            echo "ERROR: baseline failed (no '$metric_key:' in run.log). Last 20 lines:"
            tail -20 "$exp_dir/run.log" 2>/dev/null || true
            exit 1
        fi
        echo "$score" > "$best_score_file"
        local extras=()
        for k in "${extra_keys[@]}"; do extras+=("$(extract_metric "$exp_dir/run.log" "$k")"); done
        log_result "0" "$score" "keep" "baseline" "${extras[@]}"
        log_run_end "0" "$score" "keep" "baseline"
        echo "Baseline: $metric_key=$score"
    fi

    # --- Main loop ---
    while true; do
        local num; num=$(get_next_num)

        # max=0 (default) means unlimited; positive N caps post-baseline experiments.
        if [[ "$max" -gt 0 && "$num" -gt "$max" ]]; then
            echo "Max experiments ($max) reached. Stopping."
            break
        fi

        local exp_dir="$exp_root/experiment_$num"
        mkdir -p "$exp_dir"

        local best_score; best_score=$(cat "$best_score_file")
        cp "$best_train" "$train_py"
        cp "$best_train" "$exp_dir/train_before.py"
        cp "$prompt_file" "$exp_dir/prompt.md"

        local history; history="$(cat "$results_file")"
        local recent_reports=""
        local report
        for report in $(find "$exp_root" -name "report.md" -path "*/experiment_*/report.md" | sort -V | tail -5); do
            local exp_name; exp_name=$(basename "$(dirname "$report")")
            recent_reports+="
### $exp_name
$(cat "$report")
"
        done

        echo ""
        echo "========================================="
        echo "  [$project] Experiment #$num"
        echo "  Best $metric_key: $best_score ($direction)"
        echo "  $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================="

        log_run_start "$num" "$best_score"

        local run_logs_content=""
        [[ -f "$run_logs" ]] && run_logs_content="$(cat "$run_logs")"

        # Phase 1: Claude edits train.py (never runs training).
        local claude_prompt
        claude_prompt="$(cat <<PROMPT_EOF
Tu es un chercheur autonome en deep learning.

## Tes instructions
$(cat "$prompt_file")

## Contexte
- Projet : $project
- Expérience #$num
- Métrique : $metric_key ($direction — $( [[ "$direction" == "minimize" ]] && echo "plus bas = meilleur" || echo "plus haut = meilleur" ))
- Meilleur $metric_key actuel : $best_score
- Répertoire du projet : $pdir
- Dossier de cette expérience : $exp_dir

## Historique des résultats
$history

## Rapports récents
$recent_reports

## Journal des runs (run_logs.md)
$run_logs_content

## Règles strictes
- Tu ne modifies QUE $pdir/train.py (prepare.py est read-only)
- Pas de nouvelles dépendances (uniquement ce qui est dans pyproject.toml)
- Tu ne lances PAS l'entraînement toi-même — c'est ralph.sh qui s'en charge après toi
- NE PAS exécuter $train_cmd. NE PAS lancer de run.

## À faire
1. Lis $pdir/train.py et $pdir/prepare.py pour comprendre le code actuel
2. Analyse l'historique et le journal pour éviter de répéter des échecs
3. Choisis UNE idée d'amélioration (PRIORITÉ : architecturales > optimisation code > hyperparamètres)
4. Écris ce que tu vas essayer dans $run_logs (append). Inclus :
   - L'idée en une phrase
   - Pourquoi tu penses que ça va marcher
   - Les risques (crash, OOM, régression)
5. Modifie $pdir/train.py avec ton changement
6. Vérifie la syntaxe : uv run python -c "import ast; ast.parse(open('$pdir/train.py').read())"
7. Écris un rapport AVANT run dans $exp_dir/report.md :
   - Ligne 1 : description courte (UNE phrase, sans markdown heading)
   - Ce que tu as changé et pourquoi
8. Si tu as des notes pour les prochaines itérations, modifie $prompt_file

IMPORTANT : ton SEUL travail est de modifier train.py intelligemment. L'entraînement sera lancé automatiquement après.
Commence maintenant. Ne demande pas confirmation.
PROMPT_EOF
)"

        local claude_log="$exp_dir/claude_output.log"
        claude -p "$claude_prompt" --dangerously-skip-permissions > "$claude_log" 2>&1 &
        CHILD_PID=$!
        wait "$CHILD_PID" 2>/dev/null
        CHILD_PID=0
        [[ "$STOPPING" -eq 1 ]] && exit 0
        tail -5 "$claude_log" 2>/dev/null || true

        # Syntax check before running.
        if ! uv run python -c "import ast; ast.parse(open('$train_py').read())" 2>/dev/null; then
            echo ">>> SYNTAX ERROR in train.py — skipping run"
            cp "$best_train" "$train_py"
            local desc="syntax error"
            [[ -f "$exp_dir/report.md" ]] && desc=$(head -1 "$exp_dir/report.md" | sed 's/^#* *//' | tr '\t' ' ' | cut -c1-120)
            local crash_extras=(); for k in "${extra_keys[@]}"; do crash_extras+=("0"); done
            log_result "$num" "0.000000" "crash" "$desc (syntax error)" "${crash_extras[@]}"
            log_run_end "$num" "0.000000" "crash" "$desc (syntax error)"
            echo "========================================="
            echo ""
            continue
        fi

        cp "$train_py" "$exp_dir/train.py"

        # Phase 2: run training.
        echo "  Running training..."
        local run_log="$exp_dir/run.log"
        ( cd "$pdir" && timeout "$timeout_s" bash -c "$train_cmd" ) > "$run_log" 2>&1 &
        CHILD_PID=$!
        wait "$CHILD_PID" 2>/dev/null
        CHILD_PID=0
        [[ "$STOPPING" -eq 1 ]] && exit 0

        local score; score="$(extract_metric "$run_log" "$metric_key")"
        local run_extras=()
        for k in "${extra_keys[@]}"; do run_extras+=("$(extract_metric "$run_log" "$k")"); done

        local desc="experiment $num"
        [[ -f "$exp_dir/report.md" ]] && desc=$(head -1 "$exp_dir/report.md" | sed 's/^#* *//' | tr '\t' ' ' | cut -c1-120)

        if [[ -n "$score" ]]; then
            local improved; improved=$(is_improved "$score" "$best_score")
            if [[ -f "$exp_dir/report.md" ]]; then
                printf "\n\n## Résultats\n- %s: %s\n" "$metric_key" "$score" >> "$exp_dir/report.md"
                local i=0
                for k in "${extra_keys[@]}"; do
                    printf -- "- %s: %s\n" "$k" "${run_extras[$i]:-}" >> "$exp_dir/report.md"
                    i=$((i + 1))
                done
                printf -- "- status: %s\n" "$( [[ "$improved" == "yes" ]] && echo "IMPROVED" || echo "discarded" )" >> "$exp_dir/report.md"
            fi

            if [[ "$improved" == "yes" ]]; then
                cp "$train_py" "$best_train"
                echo "$score" > "$best_score_file"
                log_result "$num" "$score" "keep" "$desc" "${run_extras[@]}"
                log_run_end "$num" "$score" "keep" "$desc"
                echo ">>> IMPROVED: $best_score -> $score <<<"
            else
                cp "$best_train" "$train_py"
                log_result "$num" "$score" "discard" "$desc" "${run_extras[@]}"
                log_run_end "$num" "$score" "discard" "$desc"
                echo ">>> No improvement: $score (best: $best_score)"
            fi
        else
            cp "$best_train" "$train_py"
            echo "--- Last 10 lines of run.log ---"
            tail -10 "$run_log" 2>/dev/null || true
            echo "---"
            local crash_extras=(); for k in "${extra_keys[@]}"; do crash_extras+=("0"); done
            log_result "$num" "0.000000" "crash" "$desc" "${crash_extras[@]}"
            log_run_end "$num" "0.000000" "crash" "$desc"
            echo ">>> CRASH: no $metric_key found"
        fi

        echo "========================================="
        echo ""
    done
}

<<<<<<< HEAD
    CLAUDE_LOG="$EXP_DIR/claude_output.log"
    # Sauvegarder le prompt pour debug
    printf '%s' "$CLAUDE_PROMPT" > "$EXP_DIR/claude_prompt.txt"
    claude -p "$CLAUDE_PROMPT" --dangerously-skip-permissions > "$CLAUDE_LOG" 2>&1 &
    CHILD_PID=$!
    wait "$CHILD_PID" 2>/dev/null
    CHILD_PID=0
    [[ "$STOPPING" -eq 1 ]] && exit 0
    tail -5 "$CLAUDE_LOG" 2>/dev/null || true
=======
# --- Subcommand dispatch ---
usage() {
    cat <<EOF
Usage:
  $0 list
  $0 new <name>
  $0 run <project> [--max N]
EOF
    exit 1
}
>>>>>>> 6889b79 (feat(ralph): rewrite as subcommand dispatcher with list + new)

[[ $# -lt 1 ]] && usage
SUBCMD="$1"; shift

<<<<<<< HEAD
    # Sauvegarder la version de train.py avant le run
    cp "$ROOT/train.py" "$EXP_DIR/train.py"

    # --- Phase 2 : ralph lance l'entraînement (avec retry sur crash infra) ---
    MAX_RETRIES=3
    RETRY=0
    VAL_BPB=""
    PEAK_VRAM=""
    while [[ "$RETRY" -lt "$MAX_RETRIES" ]]; do
        echo "  Running training... (attempt $((RETRY+1))/$MAX_RETRIES)"
        RUN_LOG="$EXP_DIR/run.log"
        run_training "$RUN_LOG" &
        CHILD_PID=$!
        wait "$CHILD_PID" 2>/dev/null
        TRAIN_RC=$?
        CHILD_PID=0
        [[ "$STOPPING" -eq 1 ]] && exit 0

        VAL_BPB=$(extract_metric "$RUN_LOG" "val_bpb")
        PEAK_VRAM=$(extract_metric "$RUN_LOG" "peak_vram_mb")

        # If we got a val_bpb, or it's a legit code crash (not infra), stop retrying
        if [[ -n "$VAL_BPB" && "$VAL_BPB" != "0" ]]; then
            break
        fi

        # Check if it's an infra/CUDA crash (worth retrying) vs a code bug (not worth retrying)
        if grep -qE "CUDA driver initialization failed|CUDA out of memory|NCCL|device-side assert|illegal memory access" "$RUN_LOG" 2>/dev/null; then
            RETRY=$((RETRY + 1))
            if [[ "$RETRY" -lt "$MAX_RETRIES" ]]; then
                echo "  >>> Infra crash detected, retrying in 10s..."
                sleep 10
                continue
            fi
        fi
        break
    done

    DESC="experiment $NUM"
    if [[ -f "$EXP_DIR/report.md" ]]; then
        DESC=$(head -1 "$EXP_DIR/report.md" | sed 's/^#* *//' | tr '\t' ' ' | cut -c1-120)
    fi

    if [[ -n "$VAL_BPB" && "$VAL_BPB" != "0" ]]; then
        evaluate_result "$NUM" "$VAL_BPB" "$PEAK_VRAM" "$BEST_SCORE" "$DESC"
    else
        # --- Phase 2b : Crash analysis ---
        echo "--- Last 10 lines of run.log ---"
        tail -10 "$RUN_LOG" 2>/dev/null || true
        echo "---"
        echo ">>> CRASH detected — launching crash analysis..."

        CRASH_RESOLVED=0
        FIX_ATTEMPTS=0
        INFRA_RETRIES=0
        MAX_CRASH_INFRA_RETRIES=2

        while [[ "$CRASH_RESOLVED" -eq 0 ]]; do
            CLASSIFICATION=$(run_crash_analysis "$EXP_DIR")
            echo "  Crash classification: $CLASSIFICATION"

            if [[ "$CLASSIFICATION" == "INFRA" ]]; then
                INFRA_RETRIES=$((INFRA_RETRIES + 1))
                if [[ "$INFRA_RETRIES" -le "$MAX_CRASH_INFRA_RETRIES" ]]; then
                    echo "  >>> INFRA crash — retrying training ($INFRA_RETRIES/$MAX_CRASH_INFRA_RETRIES)..."
                    sleep 15
                    cp "$EXP_DIR/train.py" "$ROOT/train.py"
                    run_training "$RUN_LOG" &
                    CHILD_PID=$!
                    wait "$CHILD_PID" 2>/dev/null
                    CHILD_PID=0
                    [[ "$STOPPING" -eq 1 ]] && exit 0

                    VAL_BPB=$(extract_metric "$RUN_LOG" "val_bpb")
                    PEAK_VRAM=$(extract_metric "$RUN_LOG" "peak_vram_mb")

                    if [[ -n "$VAL_BPB" && "$VAL_BPB" != "0" ]]; then
                        CRASH_RESOLVED=1
                        break
                    fi
                    continue
                else
                    echo "  >>> Max INFRA retries reached — skipping"
                    break
                fi

            elif [[ "$CLASSIFICATION" == "FIX" ]]; then
                FIX_ATTEMPTS=$((FIX_ATTEMPTS + 1))
                if [[ "$FIX_ATTEMPTS" -le "$MAX_FIX_ATTEMPTS" ]]; then
                    # Vérifier la syntaxe du fix
                    if ! uv run python -c "import ast; ast.parse(open('$ROOT/train.py').read())" 2>/dev/null; then
                        echo "  >>> FIX produced invalid syntax — skipping"
                        break
                    fi
                    cp "$ROOT/train.py" "$EXP_DIR/train_fix${FIX_ATTEMPTS}.py"
                    cp "$ROOT/train.py" "$EXP_DIR/train.py"
                    echo "  >>> Code fix applied — retrying training (fix attempt $FIX_ATTEMPTS/$MAX_FIX_ATTEMPTS)..."
                    run_training "$RUN_LOG" &
                    CHILD_PID=$!
                    wait "$CHILD_PID" 2>/dev/null
                    CHILD_PID=0
                    [[ "$STOPPING" -eq 1 ]] && exit 0

                    VAL_BPB=$(extract_metric "$RUN_LOG" "val_bpb")
                    PEAK_VRAM=$(extract_metric "$RUN_LOG" "peak_vram_mb")

                    if [[ -n "$VAL_BPB" && "$VAL_BPB" != "0" ]]; then
                        CRASH_RESOLVED=1
                        break
                    fi
                    continue
                else
                    echo "  >>> Max FIX attempts reached — skipping"
                    break
                fi

            else
                # SKIP ou UNKNOWN
                echo "  >>> Crash classified as $CLASSIFICATION — skipping experiment"
                break
            fi
        done

        if [[ "$CRASH_RESOLVED" -eq 1 && -n "$VAL_BPB" && "$VAL_BPB" != "0" ]]; then
            evaluate_result "$NUM" "$VAL_BPB" "$PEAK_VRAM" "$BEST_SCORE" "$DESC" " (after crash fix)"
        else
            # Crash non résolu — revert et log
            cp "$BEST_TRAIN" "$ROOT/train.py"
            CRASH_NOTE=""
            if [[ -f "$EXP_DIR/crash_analysis.md" ]]; then
                CRASH_NOTE=" | $(head -1 "$EXP_DIR/crash_analysis.md")"
            fi
            log_result "$NUM" "0.000000" "0" "crash" "$DESC$CRASH_NOTE"
            log_run_end "$NUM" "0.000000" "crash" "$DESC$CRASH_NOTE"
            echo ">>> CRASH (unresolved): no val_bpb found"
        fi
    fi

    echo "========================================="
    echo ""
done
=======
case "$SUBCMD" in
    list) cmd_list "$@" ;;
    new)  cmd_new "$@" ;;
    run)  cmd_run "$@" ;;
    *)    usage ;;
esac
>>>>>>> 6889b79 (feat(ralph): rewrite as subcommand dispatcher with list + new)
