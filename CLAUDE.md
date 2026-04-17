# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Autonomous LLM pretraining research by Karpathy. An AI agent iteratively modifies `train.py`, runs 5-minute training experiments on a single GPU, and keeps changes that lower `val_bpb` (validation bits per byte). The agent operates autonomously in a loop — no human interaction needed once started.

## Commands

```bash
uv sync                                 # install deps
uv run projects/<name>/prepare.py       # one-time per project: data + tokenizer
./ralph.sh list                         # show projects and their best score
./ralph.sh new <name>                   # scaffold projects/<name>/ from _template
./ralph.sh run <project> [--max N]      # autonomous RALPH loop
```

Data/tokenizer caches land wherever each project's `prepare.py` decides (the default `gpt-bpb` project uses `~/.cache/autoresearch/`).

## Architecture

The repo hosts multiple research projects under `projects/<name>/`. Each project is self-contained:

- **`projects/<name>/project.toml`** — declares `metric.key`, `metric.direction` (minimize/maximize), `metric.extra_keys`, `run.train_cmd`, `run.timeout_s`. Read by `ralph.sh` to know how to execute and score the project.
- **`projects/<name>/prepare.py`** — READ-ONLY for the agent. One-time data download + tokenizer + runtime helpers (dataloader, eval function). Imported by `train.py`.
- **`projects/<name>/train.py`** — THE ONLY FILE THE AGENT EDITS. Model + optimizer + training loop. Must print `<metric.key>: <float>` (and each `extra_keys[i]: <float>`) on stdout/stderr.
- **`projects/<name>/prompt.md`** — agent instructions for this project. Edited by humans.
- **`ralph.sh`** — root orchestrator. Subcommands `run`, `new`, `list`. Never touches `prepare.py` or `project.toml`; owns `train_best.py`, `experiments/`, `run_logs.md`.

## Experiment Loop Protocol (per project)

1. `./ralph.sh run <project>` — ralph runs the loop, you never invoke it manually per experiment.
2. First run (if no `best_score`) is always the baseline, using `train_best.py`.
3. Loop: ralph invokes Claude → Claude edits `projects/<project>/train.py` → ralph runs `train_cmd` → ralph extracts `metric.key`, compares per `direction`, promotes or reverts.
4. Every experiment is logged to `projects/<project>/experiments/results.tsv`.
5. Training is hard-killed at `project.toml [run].timeout_s`.
6. Never stop to ask — run autonomously until interrupted.

## Key Constraints

- Only `projects/<name>/train.py` may be modified — `prepare.py` and `project.toml` are frozen for the agent.
- No new dependencies — only what's in the root `pyproject.toml`.
- Wall-clock budget is set per project in `project.toml [run].timeout_s`.
- Metric is per project (`project.toml [metric].key`, `[metric].direction`).
- Simplicity criterion: prefer simpler code at equal performance; reject tiny gains that add complexity.
- VRAM is a soft constraint — modest increases OK for meaningful gains.

## results.tsv Format

Tab-separated, one file per project at `projects/<name>/experiments/results.tsv`. Columns are generated from `project.toml`:

`experiment`, `<metric.key>`, `<extra_keys...>`, `status`, `description`

For `gpt-bpb`: `experiment`, `val_bpb`, `peak_vram_mb`, `status`, `description`.

`status` values: `keep` | `discard` | `crash`.

## Model Architecture Overview

GPT with: RMSNorm, rotary embeddings, Flash Attention 3, sliding window pattern (SSSL), value embeddings (ResFormer), per-layer residual/x0 lambdas, ReluSquared MLP activation, logit soft-capping. Optimizer uses Muon (polar express orthogonalization + NorMuon variance reduction) for matrix params and AdamW for embeddings/scalars.

## Tunable Hyperparameters (module-level constants in train.py)

`DEPTH`, `ASPECT_RATIO`, `HEAD_DIM`, `WINDOW_PATTERN`, `TOTAL_BATCH_SIZE`, `DEVICE_BATCH_SIZE`, `EMBEDDING_LR`, `UNEMBEDDING_LR`, `MATRIX_LR`, `SCALAR_LR`, `WEIGHT_DECAY`, `ADAM_BETAS`, `WARMUP_RATIO`, `WARMDOWN_RATIO`, `FINAL_LR_FRAC`.
