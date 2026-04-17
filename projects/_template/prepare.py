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
