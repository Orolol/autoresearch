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
