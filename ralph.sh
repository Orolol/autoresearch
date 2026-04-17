#!/usr/bin/env bash
# ralph.sh — Multi-project autoresearch orchestrator.
#
# Subcommands:
#   ./ralph.sh list                                            # list projects and their current best score
#   ./ralph.sh new <name>                                      # scaffold projects/<name>/ from projects/_template/
#   ./ralph.sh run <project> [--max N] [--remote] [--gpu TYPE] # run the RALPH loop for <project>
#
# --remote ships the project's train.py + prepare.py to RunPod Flash via train_remote.py.
# Requires RUNPOD_API_KEY in the environment.
#
# Ctrl+C stops cleanly (kills the current child process tree).

set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECTS_ROOT="$ROOT/projects"
STOPPING=0
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

require_cmd() {
    command -v "$1" &>/dev/null || { echo "ERROR: '$1' not found."; exit 1; }
}

# read_toml_str <file> <python-expr-returning-str>
read_toml_str() {
    local file="$1" expr="$2"
    uv run python - "$file" <<PY
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib
with open(sys.argv[1], "rb") as f:
    data = tomllib.load(f)
print($expr)
PY
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

require_cmd uv

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
    sed -i "s/^name = \"_template\"$/name = \"$name\"/" "$target/project.toml"

    cat <<EOF
Scaffolded: $target

Next steps:
  1. Edit $target/project.toml (metric.key, direction, extra_keys)
  2. Implement $target/prepare.py (data + tokenizer)
  3. Implement $target/train.py (training loop; print <metric.key>: <float>)
  4. Fill $target/prompt.md (hardware, exploration priorities, philosophy)

When ready: $0 run $name
EOF
}

cmd_run() {
    local project="" max=0 use_remote=0 remote_gpu="RTX_5090"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --max)    max="$2"; shift 2 ;;
            --remote) use_remote=1; shift ;;
            --gpu)    remote_gpu="$2"; shift 2 ;;
            -h|--help)
                echo "Usage: $0 run <project> [--max N] [--remote] [--gpu TYPE]"; exit 0 ;;
            -*) echo "Unknown flag: $1"; exit 1 ;;
            *)  [[ -n "$project" ]] && { echo "Unexpected arg: $1"; exit 1; }
                project="$1"; shift ;;
        esac
    done
    [[ -z "$project" ]] && { echo "Usage: $0 run <project> [--max N] [--remote] [--gpu TYPE]"; exit 1; }

    local pdir="$PROJECTS_ROOT/$project"
    local toml="$pdir/project.toml"
    [[ -f "$toml" ]] || { echo "ERROR: $toml not found."; exit 1; }

    require_cmd claude
    if [[ "$use_remote" -eq 1 ]] && [[ -z "${RUNPOD_API_KEY:-}" ]]; then
        echo "ERROR: RUNPOD_API_KEY is required when using --remote."
        exit 1
    fi

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

    # --- Training runner (local or remote) ---
    # Remote path shells out to train_remote.py which ships train.py + prepare.py to
    # RunPod Flash. The worker enforces its own timeouts; the outer `timeout` here is
    # a safety net (timeout_s + 600s to cover RunPod cold start + prepare time).
    run_training() {
        local log_file="$1"
        if [[ "$use_remote" -eq 1 ]]; then
            RUNPOD_GPU="$remote_gpu" timeout $((timeout_s + 600)) \
                uv run "$ROOT/train_remote.py" --project "$project" --gpu "$remote_gpu" \
                > "$log_file" 2>&1
        else
            ( cd "$pdir" && timeout "$timeout_s" bash -c "$train_cmd" ) > "$log_file" 2>&1
        fi
    }

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
        if [[ "$use_remote" -eq 1 ]]; then
            echo "Running remotely on $remote_gpu (timeout=${timeout_s}s) ..."
        else
            echo "Running: $train_cmd (cwd=$pdir, timeout=${timeout_s}s) ..."
        fi
        run_training "$exp_dir/run.log" &
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

        echo ""
        echo "========================================="
        echo "  [$project] Experiment #$num"
        echo "  Best $metric_key: $best_score ($direction)"
        echo "  $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================="

        log_run_start "$num" "$best_score"

        # Phase 1: Claude edits train.py (never runs training).
        #
        # The prompt embeds ONLY shell-computed scalars. Agent-authored files
        # (prompt.md, run_logs.md, reports, results.tsv) are NOT interpolated
        # — Claude reads them itself. Embedding them here would expand backticks
        # and $(...) via the heredoc (unquoted delimiter), breaking the prompt.
        local direction_hint
        if [[ "$direction" == "minimize" ]]; then
            direction_hint="plus bas = meilleur"
        else
            direction_hint="plus haut = meilleur"
        fi

        local claude_prompt
        claude_prompt="$(cat <<PROMPT_EOF
Tu es un chercheur autonome en deep learning. Expérience #$num du projet "$project".

## Contexte
- Métrique : $metric_key ($direction — $direction_hint)
- Meilleur $metric_key actuel : $best_score
- Répertoire du projet : $pdir
- Dossier de cette expérience : $exp_dir

## Étape 1 : lis ces fichiers pour avoir le contexte
- $prompt_file          (tes instructions complètes pour ce projet)
- $pdir/train.py        (le code à modifier)
- $pdir/prepare.py      (read-only, pour comprendre les données)
- $results_file         (historique TSV des résultats)
- $run_logs             (journal des runs précédents — lis au moins les 100 dernières lignes)
- Les 5 derniers rapports : \`ls -t $exp_root/experiment_*/report.md | head -5\`

## Règles strictes
- Tu ne modifies QUE $pdir/train.py (prepare.py est read-only).
- Pas de nouvelles dépendances (uniquement ce qui est dans pyproject.toml).
- Tu ne lances PAS l'entraînement toi-même — c'est ralph.sh qui s'en charge après toi.
- NE PAS exécuter "$train_cmd". NE PAS lancer de run.

## À faire
1. Lis les fichiers listés ci-dessus.
2. Analyse l'historique et le journal pour éviter de répéter des échecs.
3. Choisis UNE idée d'amélioration (PRIORITÉ : architecturales > optimisation code > hyperparamètres).
4. Append dans $run_logs :
   - L'idée en une phrase
   - Pourquoi tu penses que ça va marcher
   - Les risques (crash, OOM, régression)
5. Modifie $pdir/train.py avec ton changement.
6. Vérifie la syntaxe : \`uv run python -c "import ast; ast.parse(open('$pdir/train.py').read())"\`
7. Écris un rapport AVANT run dans $exp_dir/report.md :
   - Ligne 1 : description courte (UNE phrase, sans markdown heading)
   - Ce que tu as changé et pourquoi
8. Si tu as des notes pour les prochaines itérations, modifie $prompt_file.

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
        run_training "$run_log" &
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

# --- Subcommand dispatch ---
usage() {
    cat <<EOF
Usage:
  $0 list
  $0 new <name>
  $0 run <project> [--max N] [--remote] [--gpu TYPE]
EOF
    exit 1
}

[[ $# -lt 1 ]] && usage
SUBCMD="$1"; shift

case "$SUBCMD" in
    list) cmd_list "$@" ;;
    new)  cmd_new "$@" ;;
    run)  cmd_run "$@" ;;
    *)    usage ;;
esac
