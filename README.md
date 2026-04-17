# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `prompt.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `prompt.md` in each project is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## Features (RALPH fork)

This fork adds **RALPH** (Research Agent Loop for Persistent Hyperoptimization) — a fully autonomous experiment orchestrator on top of the original autoresearch setup.

### RALPH orchestrator (`ralph.sh`)

- **Fully autonomous experiment loop** — launches Claude in a loop to propose, implement, train, and evaluate experiments without human intervention. Just `./ralph.sh` and go to sleep.
- **Automatic crash analysis & recovery** — when a training run crashes, RALPH launches a second Claude instance to diagnose the crash (INFRA / FIX / SKIP classification). Infra crashes are retried, code bugs are auto-fixed and re-run.
- **Best-model tracking** — keeps a `train_best.py` as the current champion. Improvements are kept, regressions are automatically reverted.
- **Experiment history** — each experiment gets its own directory with the code snapshot (before & after), training logs, Claude's reasoning, and crash analyses. Results are logged to `results.tsv`.
- **Run journal** (`run_logs.md`) — human-readable log of every experiment with timestamps, status, and descriptions.
- **Remote GPU support** — run training on RunPod cloud GPUs with `./ralph.sh --remote --gpu RTX_5090`.
- **Configurable limits** — `./ralph.sh --max 20` to cap the number of experiments.
- **Graceful shutdown** — Ctrl+C cleanly kills all child processes (Claude, training, etc.).

### Research prompt (`prompt.md`)

- **Rich experiment knowledge base** — the prompt accumulates meta-lessons from 190+ experiments: what works, what doesn't, dead ends to avoid, and the current priority queue of research directions.
- **Hardware-aware** — prompt includes GPU specs (RTX 5090 / H100), CUDA version, and known platform quirks (e.g. FP8 broken on Blackwell).
- **Research philosophy** — enforces bold architectural experiments over timid hyperparameter tweaks, with automatic escalation rules.

### Architecture improvements in `train.py`

The current best model (val_bpb **1.049**, down from ~1.10 baseline) includes discoveries from 190+ automated experiments:

- **Hybrid local/global attention** — middle layers use chunkwise linear attention with learned per-head exponential decay (RetNet/GLA-inspired), bookend layers use full softmax SDPA. Best of both worlds: O(1) memory per chunk + sharp global attention where it matters.
- **RetNet-style intra-chunk attention** — softmax causal attention with learned log-decay positional bias (ALiBi-style) inside chunks, linear recurrence across chunks.
- **Non-uniform MLP expansion** — early layers get 1.5x expansion, late layers get 4.5x. Same total FLOPs as uniform 3x, better performance.
- **Value Embedding (VE) bookend** — first and last layers get shared value embeddings with learned per-head gating (ResFormer-inspired).
- **U-Net skip connections** — early layer outputs are fed to mirror late layers via learned skip lambdas.
- **Multi-scale RoPE** — per-head base frequencies (2500, 10000, 40000, 160000) for multi-resolution positional encoding.
- **RWKV-style token shift** — learnable per-channel mixing of current and previous token for K/V inputs.
- **Sparse attention gate** — per-head output gating from a small subset of input dimensions (near-square matrix for Muon optimizer).
- **Dynamic attention temperature** — input-dependent per-head Q scaling after QK-norm.
- **Logit softcap** — `tanh(logits/13)*13` for gradient flow stability.
- **Z-loss** — PaLM-style partition function penalty (`1e-4`) for logit stability.
- **x0 residual injection** — per-layer learned mixing of normalized embedding into the residual stream.
- **MuonAdamW optimizer** — Muon (polar express orthogonalization + NorMuon variance reduction) for matrix params, AdamW for embeddings/scalars. Fully `torch.compile`-friendly with fused kernels.
- **Cautious weight decay** — Muon applies weight decay only where gradient and parameter signs agree.
- **Linear warmdown schedule** — 70% of training time spent in LR cooldown to a 10% floor.

## How it works

Each research project lives under `projects/<name>/`. For a project, four files matter:

- **`project.toml`** — declares the metric name, optimization direction (`minimize`/`maximize`), the training command, and the timeout. Read by `ralph.sh`.
- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`prompt.md`** — agent instructions and accumulated research knowledge. **This file is edited and iterated on by both the human and the agent**.
- **`ralph.sh`** — autonomous experiment orchestrator. Runs the full propose → train → evaluate → keep/revert loop in a bash script. **This is the main entry point for autonomous research**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100 and RTX 5090), Python 3.10+, [uv](https://docs.astral.sh/uv/), [Claude Code](https://claude.com/claude-code).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run projects/gpt-bpb/prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run projects/gpt-bpb/train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

<<<<<<< HEAD
## Running the agent

### Option 1: RALPH (fully autonomous)

```bash
# Run experiments indefinitely (Ctrl+C to stop)
./ralph.sh

# Run at most 20 experiments
./ralph.sh --max 20

# Run on a remote RunPod GPU
./ralph.sh --remote --gpu RTX_5090
```

RALPH will:
1. Launch Claude to read the research history and propose an experiment
2. Claude modifies `train.py` with its idea
3. RALPH trains the model for 5 minutes
4. If the result improves, keep it. Otherwise, revert.
5. If training crashes, auto-diagnose and potentially fix & retry.
6. Repeat.

### Option 2: Manual (original approach)

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at prompt.md and let's kick off a new experiment! let's do the setup first.
```

The `prompt.md` file is essentially a super lightweight "skill".

=======
>>>>>>> 9720461 (docs: update README for multi-project layout)
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

`train.py` must print `<metric.key>: <float>` (and a line per entry in `extra_keys`) on stdout or stderr. The last matching line wins.

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

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.
- **Crash resilience.** RALPH automatically classifies crashes (infra vs code bug vs fundamentally broken) and retries or fixes accordingly. No wasted experiment slots.

## Platform support

This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. I'm not 100% sure that I want to take this on personally right now. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `projects/<name>/prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `projects/<name>/prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `projects/<name>/train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## License

MIT
