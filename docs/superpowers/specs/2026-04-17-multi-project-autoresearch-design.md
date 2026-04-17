# Multi-Project Autoresearch — Design

**Date:** 2026-04-17
**Status:** Approved (design phase)
**Scope:** Restructure the autoresearch repository so it can host several independent LLM/NLP research projects side by side, each with its own training code, data prep, agent prompt, experiment history, and metric definition. `ralph.sh` becomes a single dispatcher that operates on a selected project.

## Goals

- Allow N coexisting research projects, each isolated in its own folder under `projects/`.
- Preserve the current flow end-to-end for the existing project (renamed `gpt-bpb`), including its 12 experiments, `best_score`, and `train_best.py`.
- Support **different metrics per project** (not just `val_bpb`) and different optimization directions (`minimize` / `maximize`).
- Keep the orchestration simple: a single `ralph.sh` at the root with subcommands (`run`, `new`, `list`).
- Make new projects cheap to create via a `_template/` scaffold.

## Non-goals

- Supporting non-LLM domains (vision, RL, etc.) in this iteration. The design remains compatible with that direction later, but the template and docs only cover LLM/NLP projects for now.
- Sharing a common `prepare.py` across projects. Each project ships its own to keep projects fully decoupled (different datasets, tokenizers, eval functions).
- Running multiple projects concurrently. `ralph.sh run` still operates on one project at a time.
- Migrating the repo to a Python package / proper CLI. `ralph.sh` stays bash.

## Target directory layout

```
autoresearch/
├── ralph.sh                        # single orchestrator with subcommands
├── pyproject.toml                  # deps shared across projects (unchanged)
├── uv.lock
├── README.md, CLAUDE.md, progress.png
├── docs/superpowers/specs/         # design docs (this file)
└── projects/
    ├── _template/                  # empty-stub skeleton used by `ralph.sh new`
    │   ├── project.toml
    │   ├── prepare.py
    │   ├── train.py
    │   └── prompt.md
    └── gpt-bpb/                    # the current project, migrated here
        ├── project.toml
        ├── prepare.py              # (was at root, moved as-is)
        ├── train.py                # (was at root, moved as-is)
        ├── train_best.py           # (was at root, moved as-is)
        ├── prompt.md               # (was at root, moved as-is)
        ├── run_logs.md             # (was at root, moved as-is)
        ├── analysis.ipynb          # (was at root, moved as-is)
        └── experiments/
            ├── best_score
            ├── results.tsv
            └── experiment_{0..12}/
```

The repository root only keeps `ralph.sh`, `pyproject.toml` / `uv.lock`, top-level docs, and `projects/`.

## `project.toml` schema

Each project declares its runtime contract in a TOML file at the root of its folder:

```toml
name = "gpt-bpb"
description = "GPT-2 like model, minimize val_bpb on FineWeb-like data"

[metric]
key = "val_bpb"              # grep "^<key>:" in run.log to extract the score
direction = "minimize"       # "minimize" | "maximize"
extra_keys = ["peak_vram_mb"]  # also extracted and logged, but not compared

[run]
train_cmd = "uv run train.py"  # executed with cwd = projects/<name>/
timeout_s = 600                # hard kill after this many seconds
```

**Parsing rule:** `ralph.sh` reads `project.toml` once per iteration via a short inline `uv run python -c "import tomllib; ..."` and exports the fields as env vars for the rest of the script.

**Extraction rule:** for a given key `K`, the score is the last line in `run.log` matching `^K:\s*(\S+)` (same pattern as today's `extract_metric`).

**Comparison rule (`improved`):**
- `direction = "minimize"` → `new < best`
- `direction = "maximize"` → `new > best`
- No tolerance / min-delta handled by `ralph.sh`. Variance handling stays in the agent's prompt / project notes.

## `results.tsv` format

Per-project, at `projects/<name>/experiments/results.tsv`. Header is generated from `project.toml`:

```
experiment	<metric.key>	<extra_key_1>	<extra_key_2>	...	status	description
```

For `gpt-bpb` this is `experiment\tval_bpb\tpeak_vram_mb\tstatus\tdescription` — identical in spirit to today's file, so the 13 existing rows carry over unchanged modulo being re-headered if needed. **Note — unit fix for `peak_vram_mb`:** the new `ralph.sh` logs whatever `train.py` prints verbatim (MB, matching the column name). The previous orchestrator divided by 1024 before logging, so historical rows in `results.tsv` are actually in GB despite the header. New rows land in MB. This is a deliberate semantic fix — the column name was always `peak_vram_mb`, and future analysis should treat rows committed after the migration as MB.

`status` values: `keep`, `discard`, `crash` (unchanged from today).

## `ralph.sh` subcommands

One orchestrator at `./ralph.sh` with three subcommands. All paths below are relative to `projects/<project>/`.

### `./ralph.sh run <project> [--max N]`

The current main loop, with every path scoped to the selected project:

1. Verify `projects/<project>/project.toml` exists. Parse it (name, metric, direction, extra_keys, train_cmd, timeout_s).
2. Initialize if first run: create `experiments/`, write `results.tsv` header, copy `train.py` → `train_best.py` if missing, create `run_logs.md` stub if missing.
3. If `experiments/best_score` is absent, run baseline (experiment_0) using `train_best.py`.
4. Main loop (same flow as today):
   - Pick next experiment number.
   - Copy `train_best.py` → `train.py` and `train_before.py` in the experiment dir.
   - Invoke `claude -p "$PROMPT" --dangerously-skip-permissions` with the prompt assembled from: `prompt.md`, current `best_score`, paths, `results.tsv`, last 5 `report.md`, `run_logs.md`. The prompt explicitly tells Claude it's working in `projects/<project>/` and may only edit `train.py` in that project.
   - Syntax-check `train.py`; on failure, revert and log `crash`.
   - Run `timeout <timeout_s> <train_cmd>` from inside `projects/<project>/`, with output to `experiments/experiment_<N>/run.log`.
   - Extract `metric.key` + `extra_keys` from `run.log`.
   - Compare to `best_score` using `direction`. Either promote (`cp train.py train_best.py; echo score > best_score`) or revert (`cp train_best.py train.py`). Log to `results.tsv` and `run_logs.md`.
5. `--max N` caps the number of post-baseline experiments. `0` / omitted = infinite.
6. Ctrl+C trap behavior stays identical (kill child process tree, then `pkill` stragglers matching `python.*train\.py` and `claude.*dangerously`).

### `./ralph.sh new <name>`

Scaffold a new project:

1. Refuse if `projects/<name>/` already exists.
2. `cp -r projects/_template projects/<name>`.
3. In the new `project.toml`, replace `name = "_template"` with `name = "<name>"` (single-line `sed`, no other edits).
4. Print a short next-steps message pointing at the four files the user needs to fill in.

### `./ralph.sh list`

Enumerate projects and their status:

1. For each `projects/*/project.toml` (excluding `_template`):
   - Read `name`, `metric.key`, `metric.direction`.
   - Read `experiments/best_score` if present; else `—`.
   - Count `experiments/experiment_*` directories.
2. Print a plain-text table: `name | metric | direction | best | #experiments`.

## `_template/` contents

Explicitly empty stubs — not a runnable hello-world. The goal is to force the author to implement the project, not to inherit arbitrary defaults.

- **`project.toml`** — all fields present, placeholder values:
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
- **`prepare.py`** — stub with a docstring explaining what to implement (data download, tokenizer training, `make_dataloader`, `evaluate_*`). Not runnable; calling it raises `NotImplementedError`.
- **`train.py`** — stub that raises `NotImplementedError` on execution, with a module docstring listing the expected stdout contract (`<metric.key>: <float>`, each `extra_keys[i]: <float>`) and exit behavior.
- **`prompt.md`** — empty section headers (`## Hardware`, `## Pistes d'exploration`, `## Philosophie`, `## Notes pour le futur`) with short placeholder comments.

**Note:** the template is deliberately non-runnable. `./ralph.sh run <newly-scaffolded>` will fail at the baseline step until the author implements `train.py` (and, depending on its needs, `prepare.py`). This is intentional — it forces the author to own the project contract rather than inheriting a silent fallback that would corrupt the first real baseline.

## Migration

One commit that moves the current root-level project into `projects/gpt-bpb/` and introduces the new orchestrator.

**File moves (git mv):**
- `train.py`, `prepare.py`, `prompt.md`, `train_best.py`, `run_logs.md`, `analysis.ipynb` → `projects/gpt-bpb/`
- `experiments/` → `projects/gpt-bpb/experiments/`

**Deletions:**
- `program.md` — documented the old manual flow, superseded by `prompt.md` and the RALPH loop. Dropped, not migrated.

**Creations:**
- `projects/gpt-bpb/project.toml` populated with the current metric: `key = "val_bpb"`, `direction = "minimize"`, `extra_keys = ["peak_vram_mb"]`, `train_cmd = "uv run train.py"`, `timeout_s = 600`.
- `projects/_template/` with the four stub files described above.
- `docs/superpowers/specs/2026-04-17-multi-project-autoresearch-design.md` (this file).

**Rewrites:**
- `ralph.sh` — rewritten around subcommands and project scoping. The core loop body stays structurally identical to today's, but paths are parameterized on the project.
- `.gitignore` — replace the `experiments/` entry with `projects/*/experiments/`.
- `README.md` — new arborescence, new commands, explanation of `project.toml` and the `new`/`run`/`list` subcommands.
- `CLAUDE.md` — update the "Architecture (3 files that matter)" section to describe the new layout; keep the RALPH protocol description but scope file paths to `projects/<name>/`.

**Verification after migration:** `./ralph.sh list` shows `gpt-bpb` with `best = 1.086682` (or the currently recorded value) and `#experiments = 12`. A dry-run of `./ralph.sh run gpt-bpb --max 0` (which should immediately stop before running anything past baseline) confirms the paths resolve correctly.

## Interface contracts (what each piece commits to)

- **`project.toml`** → schema above is the contract between the project author and `ralph.sh`. Anything outside `[metric]` / `[run]` is free-form.
- **`train.py`** → must print `metric.key: <number>` and each `extra_keys[i]: <number>` at least once on stdout/stderr. Exit 0 on success. Anything else is treated as a crash.
- **`prepare.py`** → invoked only manually by the user (`uv run projects/<name>/prepare.py`). `ralph.sh` never calls it. One-time data/tokenizer prep only.
- **`prompt.md`** → free-form markdown consumed by the agent. `ralph.sh` just concatenates it into the Claude prompt along with history.
- **`ralph.sh`** → never edits files inside `projects/<name>/` except `train.py`, `train_best.py`, `run_logs.md`, `experiments/**`. Everything else is owned by the human or the agent.

## Risks & open questions

- **Existing `results.tsv` format compatibility** — today's file has the exact schema `experiment | val_bpb | peak_vram_mb | status | description`, which matches the new derived schema for `gpt-bpb`. No data migration needed.
- **`run_logs.md` stays per-project** — this is correct (each project's journal is meaningful only within its own timeline), but if a user wants to grep across projects they'll need to do it manually.
- **Template evolution drift** — as `gpt-bpb` evolves, `_template/` won't follow. Accepted trade-off: the template exists only to bootstrap the four files; serious new projects will likely `cp -r projects/gpt-bpb projects/<new>` and rewrite from there.
- **No concurrency guard** — running two `./ralph.sh run <same-project>` in parallel would corrupt `train_best.py` / `best_score`. Not in scope to guard against; documented in the README.

## Out of scope (possible follow-ups)

- A `project.toml` field for per-project `extra_env` / Python dependencies (currently all shared in the root `pyproject.toml`).
- Cross-project aggregated dashboards.
- Generalizing to non-LLM domains (vision, RL) — the `[metric]` block already supports it, but the template and docs would need rework.
- Rewriting `ralph.sh` in Python for better TOML parsing and argument handling.
