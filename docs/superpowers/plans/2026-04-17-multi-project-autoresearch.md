# Multi-Project Autoresearch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the repo so multiple LLM/NLP research projects live side by side under `projects/<name>/`, each with its own `project.toml` declaring the metric and direction. `ralph.sh` becomes a dispatcher with `run`, `new`, `list` subcommands.

**Architecture:** Each project is a fully self-contained directory (`train.py`, `prepare.py`, `prompt.md`, `project.toml`, `train_best.py`, `experiments/`, `run_logs.md`). A single `ralph.sh` at the repo root reads `projects/<name>/project.toml` to discover the metric key, direction, training command, and timeout, then scopes the entire RALPH loop to that project's directory. A `projects/_template/` scaffold (stubs that raise `NotImplementedError`) is used by `ralph.sh new`.

**Tech Stack:** Bash 5+, Python 3.12+ (for inline TOML parsing via `tomllib`), `uv`, `git`.

**Spec:** `docs/superpowers/specs/2026-04-17-multi-project-autoresearch-design.md`

**File structure:**
- Create: `projects/gpt-bpb/project.toml`
- Create: `projects/_template/{project.toml, prepare.py, train.py, prompt.md}`
- Create: `docs/superpowers/plans/2026-04-17-multi-project-autoresearch.md` (this file)
- Rewrite: `ralph.sh` (full rewrite from scratch)
- Modify: `.gitignore`, `README.md`, `CLAUDE.md`
- Move (git mv): `train.py`, `prepare.py`, `prompt.md`, `train_best.py`, `run_logs.md`, `analysis.ipynb`, `experiments/` → `projects/gpt-bpb/`
- Delete: `program.md` (superseded by `prompt.md`)

---

## Task 1: Migrate current project to `projects/gpt-bpb/`

**Files:**
- Move: `train.py`, `prepare.py`, `prompt.md`, `train_best.py`, `run_logs.md`, `analysis.ipynb` → `projects/gpt-bpb/`
- Move: `experiments/` → `projects/gpt-bpb/experiments/`
- Delete: `program.md`
- Create: `projects/gpt-bpb/project.toml`
- Modify: `.gitignore`

No behavior change in this task — just relocation plus the new manifest. After this task `ralph.sh` at the root is stale (refers to moved files); it will be rewritten in Task 3. Do not try to run `./ralph.sh` between Task 1 and Task 3.

- [ ] **Step 1: Create the `projects/` directory**

Run:
```bash
mkdir -p projects/gpt-bpb
```

- [ ] **Step 2: `git mv` project files into `projects/gpt-bpb/`**

Run:
```bash
git mv train.py prepare.py prompt.md train_best.py run_logs.md analysis.ipynb projects/gpt-bpb/
git mv experiments projects/gpt-bpb/experiments
```

Verify:
```bash
ls projects/gpt-bpb/
```
Expected output contains: `analysis.ipynb experiments prepare.py prompt.md run_logs.md train.py train_best.py`

- [ ] **Step 3: Delete `program.md`**

Run:
```bash
git rm program.md
```

- [ ] **Step 4: Create `projects/gpt-bpb/project.toml`**

Write file `projects/gpt-bpb/project.toml`:
```toml
name = "gpt-bpb"
description = "GPT-2 like model, minimize val_bpb on FineWeb-like data"

[metric]
key = "val_bpb"
direction = "minimize"
extra_keys = ["peak_vram_mb"]

[run]
train_cmd = "uv run train.py"
timeout_s = 600
```

- [ ] **Step 5: Update `.gitignore` to scope ralph artifacts under `projects/*/`**

Replace the existing `# Ralph experiment artifacts` block and the `results.tsv` entry. The final ignore file should end with:

```
# Results file
projects/*/experiments/results.tsv

# Ralph experiment artifacts (per-project)
projects/*/experiments/
projects/*/train_best.py
projects/*/run_logs.md
```

Concretely, edit `.gitignore`:
- Replace `results.tsv` with `projects/*/experiments/results.tsv`
- Replace `experiments/` with `projects/*/experiments/`
- Replace `train_best.py` with `projects/*/train_best.py`
- Replace `run_logs.md` with `projects/*/run_logs.md`

- [ ] **Step 6: Verify `git status` is clean of unintended changes**

Run:
```bash
git status
```

Expected:
- Renamed: six files into `projects/gpt-bpb/`
- Renamed: `experiments/...` into `projects/gpt-bpb/experiments/...`
- Deleted: `program.md`
- New: `projects/gpt-bpb/project.toml`
- Modified: `.gitignore`

No untracked experiment artifacts should appear (they're covered by the new glob patterns).

- [ ] **Step 7: Commit**

Run:
```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: migrate gpt-bpb into projects/ subdirectory

Move train.py, prepare.py, prompt.md, train_best.py, run_logs.md,
analysis.ipynb and experiments/ into projects/gpt-bpb/. Add
project.toml declaring the metric (val_bpb, minimize) and runtime
contract. Drop program.md (superseded by prompt.md under RALPH).
Scope .gitignore ralph-artifact entries under projects/*/.

ralph.sh is stale after this commit and will be rewritten in the
next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Create `projects/_template/` stubs

**Files:**
- Create: `projects/_template/project.toml`
- Create: `projects/_template/prepare.py`
- Create: `projects/_template/train.py`
- Create: `projects/_template/prompt.md`

Deliberately non-runnable — `train.py` and `prepare.py` raise `NotImplementedError`. This forces the project author to implement the contract before running `./ralph.sh run <name>`.

- [ ] **Step 1: Create `projects/_template/project.toml`**

Write file `projects/_template/project.toml`:
```toml
name = "_template"
description = "TODO: describe the project"

[metric]
key = "TODO_metric"
direction = "minimize"
extra_keys = []

[run]
train_cmd = "uv run train.py"
timeout_s = 600
```

- [ ] **Step 2: Create `projects/_template/prepare.py`**

Write file `projects/_template/prepare.py`:
```python
"""One-time data preparation for this project.

Implement this module to:
- download / cache any training and validation data,
- train or load a tokenizer,
- expose a `make_dataloader(...)` helper consumed by train.py,
- expose an `evaluate_*` function consumed by train.py for the metric
  declared in project.toml (for example `evaluate_bpb` if
  `metric.key = "val_bpb"`).

This module is invoked manually by the user (e.g.
`uv run projects/<name>/prepare.py`). ralph.sh never runs it.
"""

raise NotImplementedError(
    "projects/_template/prepare.py is a stub. "
    "Replace it with real data preparation code."
)
```

- [ ] **Step 3: Create `projects/_template/train.py`**

Write file `projects/_template/train.py`:
```python
"""Training entry point for this project.

Contract consumed by ralph.sh:
- Exit 0 on success, non-zero on failure (ralph.sh logs non-zero as
  `crash`).
- Print at least once on stdout or stderr:
    <metric.key>: <float>         # the score for this run
    <extra_key>: <float>          # once per entry in metric.extra_keys
  The last matching line wins (same convention as the gpt-bpb project).
- All paths must be relative to the project directory (this file's
  parent). ralph.sh invokes `train_cmd` with cwd set there.

ralph.sh runs this under a hard timeout (project.toml [run].timeout_s).
"""

raise NotImplementedError(
    "projects/_template/train.py is a stub. "
    "Replace it with a real training loop that prints the metric "
    "declared in project.toml."
)
```

- [ ] **Step 4: Create `projects/_template/prompt.md`**

Write file `projects/_template/prompt.md`:
```markdown
# <Project name> — Instructions pour l'agent

Tu es un chercheur autonome qui optimise <décris la tâche ici>.

## Hardware

<!-- TODO: GPU, VRAM, précision par défaut, versions clés -->

## Pistes d'exploration

<!-- TODO: lister les directions que l'agent peut explorer, par priorité -->

## Philosophie

<!-- TODO: règles de décision (audace vs micro-tuning, simplicité, etc.) -->

## Notes pour le futur

<!-- Append-only: l'agent consigne ici les leçons confirmées et les pistes restantes -->
```

- [ ] **Step 5: Verify files exist and no syntax errors in Python stubs**

Run:
```bash
uv run python -c "import ast; ast.parse(open('projects/_template/prepare.py').read()); ast.parse(open('projects/_template/train.py').read()); print('ok')"
```

Expected output: `ok`

Also verify the stubs actually raise when executed:
```bash
cd projects/_template && uv run python train.py; cd ../..
```
Expected: exits non-zero with `NotImplementedError: projects/_template/train.py is a stub. ...`.

- [ ] **Step 6: Commit**

Run:
```bash
git add projects/_template/
git commit -m "$(cat <<'EOF'
feat: add projects/_template/ skeleton for ralph.sh new

Four stubs: project.toml with placeholder metric, prepare.py and
train.py that raise NotImplementedError with a documented stdout
contract, and an empty prompt.md with the standard section headers.

The template is intentionally non-runnable so authors implement the
contract before the first baseline.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Rewrite `ralph.sh` — shared helpers + `list` + `new` subcommands

**Files:**
- Rewrite: `ralph.sh` (full rewrite, starts with helpers + `list` + `new`; `run` arrives in Task 4)

The rewrite starts from scratch. The old script is 414 lines of hardcoded root-level paths; a rewrite is cheaper than adapting it. We land `list` and `new` first (they are simple, read-only or template-copy) to validate the TOML helper and argument parsing before touching the main loop.

- [ ] **Step 1: Replace `ralph.sh` with the new skeleton (helpers + subcommand dispatch)**

Overwrite `ralph.sh` with:
```bash
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

# --- Prérequis globaux ---
require_cmd() {
    command -v "$1" &>/dev/null || { echo "ERROR: '$1' not found."; exit 1; }
}
require_cmd uv

# --- TOML helpers (inline Python, tomllib is stdlib in 3.11+) ---
# read_toml_str <file> <python-expr-returning-str>
read_toml_str() {
    local file="$1" expr="$2"
    uv run python - "$file" <<PY
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    data = tomllib.load(f)
print($expr)
PY
}

# read_toml_list <file> <python-expr-returning-iterable>
read_toml_list() {
    local file="$1" expr="$2"
    uv run python - "$file" <<PY
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    data = tomllib.load(f)
for x in $expr:
    print(x)
PY
}

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

[[ $# -lt 1 ]] && usage
SUBCMD="$1"; shift

case "$SUBCMD" in
    list) cmd_list "$@" ;;
    new)  cmd_new "$@" ;;
    run)  cmd_run "$@" ;;
    *)    usage ;;
esac
```

Functions `cmd_list`, `cmd_new`, `cmd_run` are declared in later steps. The dispatch `case` references them — bash resolves function calls at call time (not parse time), so the script parses fine even while they're missing. We won't execute the script until `cmd_list` and `cmd_new` exist.

- [ ] **Step 2: Add `cmd_list` above the dispatch block**

Insert right after the `usage()` function, before the `[[ $# -lt 1 ]]` line:

```bash
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

    [[ "$any" -eq 0 ]] && echo "(no projects yet; only _template exists)"
}
```

- [ ] **Step 3: Add `cmd_new` below `cmd_list`**

```bash
cmd_new() {
    [[ $# -ne 1 ]] && { echo "Usage: $0 new <name>"; exit 1; }
    local name="$1"

    case "$name" in
        ""|_template|*/*|*" "*)
            echo "ERROR: invalid project name '$name'."
            exit 1
            ;;
    esac

    local target="$PROJECTS_ROOT/$name"
    local template="$PROJECTS_ROOT/_template"

    [[ -e "$target" ]]     && { echo "ERROR: $target already exists."; exit 1; }
    [[ ! -d "$template" ]] && { echo "ERROR: $template not found."; exit 1; }

    cp -r "$template" "$target"
    # Replace only the name = "_template" line in the new project.toml.
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
```

- [ ] **Step 4: Add `cmd_run` stub (real implementation lands in Task 4)**

Insert below `cmd_new`, still before the dispatch block:

```bash
cmd_run() {
    echo "ERROR: 'run' subcommand not implemented yet (see Task 4)."
    exit 1
}
```

This stub keeps the dispatch table valid so `list` and `new` are usable end-of-task-3 without waiting for Task 4.

- [ ] **Step 5: Make executable, syntax-check**

Run:
```bash
chmod +x ralph.sh
bash -n ralph.sh && echo "syntax ok"
```

Expected output: `syntax ok`

- [ ] **Step 6: Smoke-test `./ralph.sh list`**

Run:
```bash
./ralph.sh list
```

Expected: a two-line header plus exactly one data row for `gpt-bpb`, something like:
```
name                      metric            direction   best            #experiments
----                      ------            ---------   ----            ------------
gpt-bpb                   val_bpb           minimize    1.089807        13
```
(`#experiments` includes `experiment_0..experiment_12` → 13. The `best` value comes from `projects/gpt-bpb/experiments/best_score`.)

- [ ] **Step 7: Smoke-test `./ralph.sh new`**

Run:
```bash
./ralph.sh new smoke-test
ls projects/smoke-test/
grep '^name = ' projects/smoke-test/project.toml
```

Expected:
- Listing shows `project.toml prepare.py prompt.md train.py`.
- `grep` outputs exactly `name = "smoke-test"`.

Clean up (the scratch project is not committed):
```bash
rm -rf projects/smoke-test
```

- [ ] **Step 8: Smoke-test that `new` refuses collisions**

Run:
```bash
./ralph.sh new gpt-bpb
```

Expected: exits non-zero with `ERROR: <path> already exists.` — `projects/gpt-bpb/` is untouched.

- [ ] **Step 9: Smoke-test `cmd_run` stub**

Run:
```bash
./ralph.sh run gpt-bpb
```

Expected: exits non-zero with `ERROR: 'run' subcommand not implemented yet (see Task 4).` — no side effects.

- [ ] **Step 10: Commit**

Run:
```bash
git add ralph.sh
git commit -m "$(cat <<'EOF'
feat(ralph): rewrite as subcommand dispatcher with list + new

Ship list (enumerate projects + best score) and new (scaffold from
projects/_template/). Adds read_toml_str / read_toml_list helpers
using stdlib tomllib via uv run python. The run subcommand is a
stub that exits with an explanatory error; implementation lands in
the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Implement `ralph.sh run <project>`

**Files:**
- Modify: `ralph.sh` (replace the `cmd_run` stub)

Core RALPH loop, scoped to the selected project. Mirrors today's logic but every path is derived from the project directory and every comparison respects `metric.direction`.

- [ ] **Step 1: Replace `cmd_run` with the full implementation**

Open `ralph.sh`. Replace the entire `cmd_run() { ... }` block (the stub from Task 3, Step 4) with:

```bash
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
```

Key changes vs. today's monolithic script:
- All paths derive from `$pdir = projects/$project/` (no root-level hardcoded paths).
- `train_cmd` and `timeout_s` come from `project.toml` (previously hardcoded to `uv run $ROOT/train.py` and `600`).
- `is_improved` is direction-aware (previously hardcoded `<`).
- `results.tsv` header is generated from `metric.key` + `extra_keys` on first init. Extras are appended dynamically per row.
- `run_logs.md` messages use `$metric_key` instead of the literal `val_bpb`.
- Train commands run with `cd "$pdir"` so relative paths inside `train.py` (e.g. `./train_best.py`, `./prepare.py`) still resolve.
- `peak_vram_mb` is no longer divided by 1024 in the script; whatever `train.py` prints lands verbatim in `results.tsv`. The existing `gpt-bpb/train.py` already prints `peak_vram_mb:` in MB — we keep the raw MB value for simplicity and let the agent's prompt explain the unit. (This is a deliberate, minor behavior change from today's script, which divided by 1024. Documented in README.)

- [ ] **Step 2: Syntax-check**

Run:
```bash
bash -n ralph.sh && echo "syntax ok"
```

Expected: `syntax ok`.

- [ ] **Step 3: Smoke-test — config loading failure path**

Create a bogus project and point `run` at it:
```bash
./ralph.sh new syntax-check
./ralph.sh run syntax-check --max 0
```

Expected: the baseline attempt invokes `uv run train.py`, which raises `NotImplementedError`, `extract_metric` finds no `TODO_metric:` in the log, and the script exits non-zero with `ERROR: baseline failed (no 'TODO_metric:' in run.log).` The last 20 lines of run.log are printed (showing the NotImplementedError traceback).

Clean up:
```bash
rm -rf projects/syntax-check
```

- [ ] **Step 4: Smoke-test — `--max` argument parsing**

Run:
```bash
./ralph.sh run nonexistent-project --max 1
```

Expected: `ERROR: <ROOT>/projects/nonexistent-project/project.toml not found.` and exit 1. Arg parsing accepts `--max 1` before failing on the project lookup.

- [ ] **Step 5: Smoke-test — usage errors**

Run each line and verify it fails usefully:
```bash
./ralph.sh run
./ralph.sh run --max 5
./ralph.sh run gpt-bpb --bogus-flag
```

Expected messages:
- `Usage: ./ralph.sh run <project> [--max N]`
- `Usage: ./ralph.sh run <project> [--max N]`
- `Unknown flag: --bogus-flag`

- [ ] **Step 6: Dry-run verification on `gpt-bpb`**

**Don't actually run training** (5 minutes). Verify that the `gpt-bpb` manifest parses exactly to the values `cmd_run` would read:

```bash
uv run python -c "
import tomllib
with open('projects/gpt-bpb/project.toml', 'rb') as f:
    d = tomllib.load(f)
print(d['metric']['key'])
print(d['metric']['direction'])
print(d['run']['train_cmd'])
print(d['run']['timeout_s'])
print(d['metric'].get('extra_keys', []))
"
```

Expected output:
```
val_bpb
minimize
uv run train.py
600
['peak_vram_mb']
```

A real `./ralph.sh run gpt-bpb --max 0` would take ~5 minutes (baseline training) and doesn't need to be executed as part of this plan — it's the user's call whether to burn the GPU time.

- [ ] **Step 7: Commit**

Run:
```bash
git add ralph.sh
git commit -m "$(cat <<'EOF'
feat(ralph): implement run subcommand scoped to project

The RALPH loop now reads project.toml for metric.key,
metric.direction, metric.extra_keys, run.train_cmd, and
run.timeout_s, then runs every phase with cwd set to
projects/<name>/. Comparison respects direction (minimize or
maximize). results.tsv header is generated from the manifest so
projects with different metrics keep a readable schema.

peak_vram_mb is no longer divided by 1024 in the logger — whatever
train.py prints lands verbatim in results.tsv.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Update `README.md`

**Files:**
- Modify: `README.md`

Reflect the new layout, the three subcommands, the `project.toml` contract, and the migration. Keep Karpathy's intro and design-choices sections intact.

- [ ] **Step 1: Replace the "Project structure" section**

Open `README.md`. Find the `## Project structure` section (currently lists `prepare.py, train.py, train_best.py, program.md, prompt.md, ralph.sh, experiments/, run_logs.md, pyproject.toml`) and replace it with:

```markdown
## Project structure

```
ralph.sh              — orchestrator (subcommands: run, new, list)
pyproject.toml        — shared deps
projects/
├── _template/        — skeleton used by `ralph.sh new` (non-runnable stubs)
└── <name>/           — one self-contained research project
    ├── project.toml      — metric, direction, train command, timeout
    ├── prepare.py        — data + tokenizer (manual, read-only for the agent)
    ├── train.py          — model + training loop (the agent edits this)
    ├── train_best.py     — current best version (managed by ralph.sh)
    ├── prompt.md         — agent instructions for this project
    ├── run_logs.md       — human-readable journal (maintained by the agent)
    └── experiments/      — per-run artifacts + results.tsv + best_score
```
```

- [ ] **Step 2: Replace the "Running autonomously with `ralph.sh`" section**

Find the `## Running autonomously with \`ralph.sh\`` section added in the previous pass and replace its body with the new subcommand documentation:

```markdown
## Running autonomously with `ralph.sh`

The repo ships a bash orchestrator that runs the experiment loop fully unattended. It dispatches on subcommands and scopes everything to a single project under `projects/`.

```bash
./ralph.sh list                         # show projects and their current best score
./ralph.sh new <name>                   # scaffold projects/<name>/ from _template
./ralph.sh run <project> [--max N]      # run the RALPH loop
```

Requirements: `claude` CLI (Claude Code) on `$PATH`, `uv`, and a project whose `train.py` actually runs. Press Ctrl+C at any time — the trap kills the current child (Claude or `train.py`) and exits cleanly.

Each project declares its runtime contract in `project.toml`:

```toml
name = "gpt-bpb"
description = "GPT-2 like model, minimize val_bpb on FineWeb-like data"

[metric]
key = "val_bpb"              # ralph.sh extracts "<key>:" from run.log
direction = "minimize"       # "minimize" or "maximize"
extra_keys = ["peak_vram_mb"]

[run]
train_cmd = "uv run train.py"
timeout_s = 600
```

`train.py` must print `<metric.key>: <float>` (and each `extra_keys[i]: <float>`) on stdout or stderr. The last matching line wins.

### What `ralph.sh run` does per iteration

1. Copy `train_best.py` → `train.py` (always start from the current best).
2. Call `claude -p` with `prompt.md` + the full project history (`results.tsv`, last 5 reports, `run_logs.md`). Claude edits `train.py` only.
3. Syntax-check `train.py`; revert and mark `crash` if invalid.
4. Run `train_cmd` from inside `projects/<project>/` with a hard timeout.
5. Extract the metric (respecting `direction`). If improved → promote to `train_best.py` and update `best_score`; otherwise revert. Log either way.

Per-experiment artifacts land in `projects/<project>/experiments/experiment_<N>/`: `train_before.py`, `train.py`, `prompt.md`, `run.log`, `claude_output.log`, `report.md`.

### Scaffolding a new project

```bash
./ralph.sh new my-project
```

The template stubs for `prepare.py` and `train.py` both raise `NotImplementedError` — intentional, so you own the contract before the first run. Fill them in, then `./ralph.sh run my-project`.
```

- [ ] **Step 3: Drop the old "Running the agent" section**

The section that instructs users to prompt Claude manually with `program.md` is obsolete (`program.md` is deleted). Remove the entire `## Running the agent` section (from the `## Running the agent` heading through the paragraph ending with `The \`program.md\` file is essentially a super lightweight "skill".`).

- [ ] **Step 4: Visual-inspect the README**

Run:
```bash
grep -n '^## ' README.md
```

Expected section list (in order): `How it works`, `Quick start`, `Running autonomously with \`ralph.sh\``, `Project structure`, `Design choices`, `Platform support`, `Notable forks`, `License`.

No `## Running the agent` section. No reference to `program.md` anywhere:
```bash
! grep -n 'program\.md' README.md && echo "clean"
```
Expected: `clean`.

- [ ] **Step 5: Commit**

Run:
```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: update README for multi-project layout

Describe the new projects/ directory structure, the three ralph.sh
subcommands (run/new/list), and the project.toml contract. Drop the
manual-agent section (program.md is gone).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md` (project-level)

Keep the RALPH protocol description but repath everything to `projects/<name>/`.

- [ ] **Step 1: Replace the "Commands" section**

Find the `## Commands` section in `CLAUDE.md` and replace its body with:

```markdown
## Commands

```bash
uv sync                                 # install deps
uv run projects/<name>/prepare.py       # one-time per project: data + tokenizer
./ralph.sh list                         # show projects and their best score
./ralph.sh new <name>                   # scaffold projects/<name>/ from _template
./ralph.sh run <project> [--max N]      # autonomous RALPH loop
```

Data/tokenizer caches land wherever each project's `prepare.py` decides (the default `gpt-bpb` project uses `~/.cache/autoresearch/`).
```

- [ ] **Step 2: Replace the "Architecture (3 files that matter)" section**

Find `## Architecture (3 files that matter)` and replace it with:

```markdown
## Architecture

The repo hosts multiple research projects under `projects/<name>/`. Each project is self-contained:

- **`projects/<name>/project.toml`** — declares `metric.key`, `metric.direction` (minimize/maximize), `metric.extra_keys`, `run.train_cmd`, `run.timeout_s`. Read by `ralph.sh` to know how to execute and score the project.
- **`projects/<name>/prepare.py`** — READ-ONLY for the agent. One-time data download + tokenizer + runtime helpers (dataloader, eval function). Imported by `train.py`.
- **`projects/<name>/train.py`** — THE ONLY FILE THE AGENT EDITS. Model + optimizer + training loop. Must print `<metric.key>: <float>` (and each `extra_keys[i]: <float>`) on stdout/stderr.
- **`projects/<name>/prompt.md`** — agent instructions for this project. Edited by humans.
- **`ralph.sh`** — root orchestrator. Subcommands `run`, `new`, `list`. Never touches `prepare.py` or `project.toml`; owns `train_best.py`, `experiments/`, `run_logs.md`.
```

- [ ] **Step 3: Replace the "Experiment Loop Protocol" section**

Find `## Experiment Loop Protocol` and replace it with:

```markdown
## Experiment Loop Protocol (per project)

1. `./ralph.sh run <project>` — ralph runs the loop, you never invoke it manually per experiment.
2. First run (if no `best_score`) is always the baseline, using `train_best.py`.
3. Loop: ralph invokes Claude → Claude edits `projects/<project>/train.py` → ralph runs `train_cmd` → ralph extracts `metric.key`, compares per `direction`, promotes or reverts.
4. Every experiment is logged to `projects/<project>/experiments/results.tsv`.
5. Training is hard-killed at `project.toml [run].timeout_s`.
6. Never stop to ask — run autonomously until interrupted.
```

- [ ] **Step 4: Replace the "Key Constraints" section**

Find `## Key Constraints` and replace with:

```markdown
## Key Constraints

- Only `projects/<name>/train.py` may be modified — `prepare.py` and `project.toml` are frozen for the agent.
- No new dependencies — only what's in the root `pyproject.toml`.
- Wall-clock budget is set per project in `project.toml [run].timeout_s`.
- Metric is per project (`project.toml [metric].key`, `[metric].direction`).
- Simplicity criterion: prefer simpler code at equal performance; reject tiny gains that add complexity.
- VRAM is a soft constraint — modest increases OK for meaningful gains.
```

- [ ] **Step 5: Replace the "results.tsv Format" section**

Find `## results.tsv Format` and replace with:

```markdown
## results.tsv Format

Tab-separated, one file per project at `projects/<name>/experiments/results.tsv`. Columns are generated from `project.toml`:

`experiment`, `<metric.key>`, `<extra_keys...>`, `status`, `description`

For `gpt-bpb`: `experiment`, `val_bpb`, `peak_vram_mb`, `status`, `description`.

`status` values: `keep` | `discard` | `crash`.
```

- [ ] **Step 6: Visual-inspect**

Run:
```bash
grep -n '^## ' CLAUDE.md
```

Confirm the expected section ordering and that no section mentions root-level `train.py` / `prepare.py` / `experiments/` paths anymore.

```bash
! grep -n '^train\.py\|^prepare\.py\| experiments/$' CLAUDE.md && echo "clean"
```
Expected: `clean`.

- [ ] **Step 7: Commit**

Run:
```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(CLAUDE.md): reflect multi-project layout

Repath architecture/protocol sections under projects/<name>/, make
metric + timeout per-project via project.toml, and drop references
to root-level train.py / prepare.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: End-to-end smoke verification

**Files:** none (verification only).

The five commits above should leave the repo in a working state. This task is a read-only checklist to confirm it before we hand off.

- [ ] **Step 1: Git state is clean**

Run:
```bash
git status
```
Expected: `nothing to commit, working tree clean`.

- [ ] **Step 2: `list` shows gpt-bpb**

Run:
```bash
./ralph.sh list
```
Expected: one data row for `gpt-bpb`, `metric = val_bpb`, `direction = minimize`, `best` matches `cat projects/gpt-bpb/experiments/best_score`, `#experiments = 13`.

- [ ] **Step 3: `new` / cleanup cycle works**

Run:
```bash
./ralph.sh new e2e-check
./ralph.sh list
rm -rf projects/e2e-check
./ralph.sh list
```
Expected: the second `list` shows two rows (`e2e-check`, `gpt-bpb`); the third shows only `gpt-bpb` again.

- [ ] **Step 4: Artifacts are correctly git-ignored**

Run:
```bash
ls projects/gpt-bpb/experiments/ | head
git check-ignore -v projects/gpt-bpb/experiments/best_score projects/gpt-bpb/experiments/results.tsv projects/gpt-bpb/train_best.py projects/gpt-bpb/run_logs.md
```
Expected: every listed file matches one of the `.gitignore` patterns — no ignored files are tracked.

- [ ] **Step 5: No dangling references**

Run:
```bash
grep -rn 'program\.md' README.md CLAUDE.md ralph.sh 2>/dev/null || echo "clean"
grep -n '\$ROOT/train\.py\|\$ROOT/prepare\.py' ralph.sh || echo "clean"
```
Expected: both print `clean`.

If all five checks pass, the migration is done and the repo is ready for `./ralph.sh run gpt-bpb` to resume the existing experiment sequence.
