# /// script
# requires-python = ">=3.10"
# dependencies = ["runpod-flash", "tomli; python_version < '3.11'"]
# ///
"""
Remote training via RunPod Flash — multi-project aware.

Ships `projects/<name>/train.py` and `projects/<name>/prepare.py` to a
RunPod Flash worker, executes them, and returns stdout/stderr exactly as
if the run happened locally.

Project config (`projects/<name>/project.toml`) drives:
- `[run].timeout_s`              — cap for train.py on the remote worker
- `[remote].deps`                — pip deps baked into the worker image
- `[remote].prepare_timeout_s`   — cap for prepare.py (default 600s)

The endpoint name is `autoresearch-train-<project>-<gpu>`, so each
(project, gpu) pair gets its own template. Different projects can declare
different deps without conflicting with each other.

Usage:
    RUNPOD_API_KEY="xxx" uv run train_remote.py --project gpt-bpb
    RUNPOD_API_KEY="xxx" uv run train_remote.py --project gpt-bpb --gpu A100
    AUTORESEARCH_PROJECT=gpt-bpb RUNPOD_GPU=A100 RUNPOD_API_KEY="xxx" uv run train_remote.py
"""

import argparse
import asyncio
import os
import pickle
import re
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

from runpod_flash import Endpoint, GpuType

# ---------------------------------------------------------------------------
# GPU type mapping (CLI name -> GpuType enum)
# ---------------------------------------------------------------------------

GPU_MAP = {
    "H100": GpuType.NVIDIA_H100_80GB_HBM3,
    "A100": GpuType.NVIDIA_A100_80GB_PCIe,
    "RTX_4090": GpuType.NVIDIA_GEFORCE_RTX_4090,
    "RTX_5090": GpuType.NVIDIA_GEFORCE_RTX_5090,
    "RTX_6000_ADA": GpuType.NVIDIA_RTX_6000_ADA_GENERATION,
    "H200": GpuType.NVIDIA_H200,
    "A40": GpuType.NVIDIA_A40,
}

DEFAULT_GPU = "RTX_5090"
DEFAULT_PREPARE_TIMEOUT_S = 600
ROOT = Path(__file__).parent
PROJECTS_ROOT = ROOT / "projects"
ENDPOINT_PREFIX = "autoresearch-train-"

# ---------------------------------------------------------------------------
# Resolve project + GPU at module load time.
#
# runpod_flash's @Endpoint decorator registers the endpoint (name, GPU, deps)
# when the module is imported. We therefore need to know the project and GPU
# before `remote_train` is defined. We do a minimal pre-parse of sys.argv +
# env vars here; the full argparse in main() re-validates the same flags.
# ---------------------------------------------------------------------------


def _preparse_project_and_gpu() -> tuple[str, str]:
    """Return (project, gpu_name). Errors out early with a helpful message."""
    project = os.environ.get("AUTORESEARCH_PROJECT")
    gpu = os.environ.get("RUNPOD_GPU", DEFAULT_GPU)

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--project" and i + 1 < len(argv):
            project = argv[i + 1]
            i += 2
        elif tok.startswith("--project="):
            project = tok.split("=", 1)[1]
            i += 1
        elif tok == "--gpu" and i + 1 < len(argv):
            gpu = argv[i + 1]
            i += 2
        elif tok.startswith("--gpu="):
            gpu = tok.split("=", 1)[1]
            i += 1
        else:
            i += 1

    if not project:
        sys.stderr.write(
            "ERROR: --project <name> is required (or set AUTORESEARCH_PROJECT).\n"
            "Available projects:\n"
        )
        if PROJECTS_ROOT.is_dir():
            for p in sorted(PROJECTS_ROOT.iterdir()):
                if p.is_dir() and p.name != "_template" and (p / "project.toml").is_file():
                    sys.stderr.write(f"  - {p.name}\n")
        sys.exit(2)

    if gpu not in GPU_MAP:
        sys.stderr.write(
            f"ERROR: unknown GPU '{gpu}'. Valid: {', '.join(GPU_MAP)}\n"
        )
        sys.exit(2)

    return project, gpu


def _load_project_config(project: str) -> dict:
    """Read projects/<project>/project.toml and validate required fields."""
    pdir = PROJECTS_ROOT / project
    toml_path = pdir / "project.toml"
    if not toml_path.is_file():
        sys.stderr.write(f"ERROR: {toml_path} not found.\n")
        sys.exit(2)
    with toml_path.open("rb") as f:
        cfg = tomllib.load(f)

    run = cfg.get("run", {})
    remote = cfg.get("remote", {})
    train_timeout = int(run.get("timeout_s", 900))
    deps = list(remote.get("deps", []))
    prepare_timeout = int(remote.get("prepare_timeout_s", DEFAULT_PREPARE_TIMEOUT_S))

    if not deps:
        sys.stderr.write(
            f"ERROR: projects/{project}/project.toml is missing [remote].deps. "
            "Add a list of pip packages the remote worker needs (e.g. "
            '["torch", "numpy"]).\n'
        )
        sys.exit(2)

    train_py = pdir / "train.py"
    prepare_py = pdir / "prepare.py"
    for path in (train_py, prepare_py):
        if not path.is_file():
            sys.stderr.write(f"ERROR: {path} not found.\n")
            sys.exit(2)

    return {
        "project": project,
        "pdir": pdir,
        "train_py_path": train_py,
        "prepare_py_path": prepare_py,
        "deps": deps,
        "train_timeout_s": train_timeout,
        "prepare_timeout_s": prepare_timeout,
    }


def _sanitize(s: str) -> str:
    """RunPod endpoint names are limited; keep [a-z0-9-] only."""
    return re.sub(r"[^a-z0-9-]+", "-", s.lower()).strip("-")


# Resolve once, at module load.
_PROJECT, _GPU_NAME = _preparse_project_and_gpu()
_CFG = _load_project_config(_PROJECT)
_GPU_TYPE = GPU_MAP[_GPU_NAME]
_ENDPOINT_NAME = f"{ENDPOINT_PREFIX}{_sanitize(_PROJECT)}-{_sanitize(_GPU_NAME)}"


# ---------------------------------------------------------------------------
# Pre-flight: sync existing RunPod endpoints into local pickle cache.
# Prevents "endpoint template names must be unique" errors when the pickle
# is missing an entry for an endpoint that already exists on RunPod.
# ---------------------------------------------------------------------------

def _sync_existing_endpoints() -> None:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        return  # Will fail later with a proper error message.

    pickle_path = ROOT / ".runpod" / "resources.pkl"

    resources: dict = {}
    hashes: dict = {}
    if pickle_path.exists():
        try:
            with pickle_path.open("rb") as f:
                resources, hashes = pickle.load(f)
        except Exception:
            pass

    try:
        from runpod_flash.core.api.runpod import RunpodGraphQLClient
        from runpod_flash.core.resources.live_serverless import LiveServerless

        async def _query() -> list[dict]:
            client = RunpodGraphQLClient(api_key=api_key)
            query = "query { myself { endpoints { id name templateId gpuIds } } }"
            result = await client._execute_graphql(query, {})
            return result["myself"]["endpoints"]

        endpoints = asyncio.run(_query())
    except Exception:
        return  # Non-fatal; will attempt a normal deploy.

    for ep in endpoints:
        ep_name = ep.get("name", "")
        if not ep_name.startswith(ENDPOINT_PREFIX):
            continue
        resource_key = f"LiveServerless:{ep_name}"
        if resource_key in resources:
            continue

        ref = next(iter(resources.values()), None) if resources else None
        try:
            if ref is not None:
                fields = {field: getattr(ref, field) for field in LiveServerless.model_fields}
            else:
                fields = {
                    field: info.default
                    for field, info in LiveServerless.model_fields.items()
                    if info.default is not None
                }
            fields["id"] = ep["id"]
            fields["name"] = ep_name
            fields["templateId"] = ep.get("templateId", "")
            fields["gpuIds"] = ep.get("gpuIds", "")
            for gpu_cli_name, gpu_type in GPU_MAP.items():
                if _sanitize(gpu_cli_name) in ep_name:
                    fields["gpus"] = [gpu_type]
                    break
            obj = LiveServerless(**fields)
            resources[resource_key] = obj
            hashes[resource_key] = obj.config_hash
            print(f"Synced existing RunPod endpoint: {ep_name} (id={ep['id']})", file=sys.stderr)
        except Exception as e:
            print(f"Warning: could not sync endpoint {ep_name}: {e}", file=sys.stderr)

    try:
        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with pickle_path.open("wb") as f:
            pickle.dump((resources, hashes), f)
    except Exception:
        pass


_sync_existing_endpoints()

# ---------------------------------------------------------------------------
# Remote endpoint: receives code as strings, writes files, runs training.
# Endpoint name is scoped to (project, gpu) so deps can differ per project.
# ---------------------------------------------------------------------------


@Endpoint(
    name=_ENDPOINT_NAME,
    gpu=_GPU_TYPE,
    workers=(0, 1),
    dependencies=_CFG["deps"],
    execution_timeout_ms=0,
)
async def remote_train(data: dict) -> dict:
    import os
    import subprocess
    import tempfile

    train_py = data["train_py"]
    prepare_py = data["prepare_py"]
    prepare_timeout = int(data.get("prepare_timeout_s", DEFAULT_PREPARE_TIMEOUT_S))
    train_timeout = int(data.get("train_timeout_s", 900))

    workdir = tempfile.mkdtemp(prefix="autoresearch_")
    train_path = os.path.join(workdir, "train.py")
    prepare_path = os.path.join(workdir, "prepare.py")

    with open(train_path, "w") as f:
        f.write(train_py)
    with open(prepare_path, "w") as f:
        f.write(prepare_py)

    prep_result = subprocess.run(
        ["python", "prepare.py"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=prepare_timeout,
    )
    if prep_result.returncode != 0:
        return {
            "stdout": prep_result.stdout,
            "stderr": f"prepare.py failed:\n{prep_result.stderr}",
            "returncode": prep_result.returncode,
        }

    train_result = subprocess.run(
        ["python", "train.py"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=train_timeout,
    )

    return {
        "stdout": train_result.stdout,
        "stderr": train_result.stderr,
        "returncode": train_result.returncode,
    }


# ---------------------------------------------------------------------------
# Local main: read project files, ship them to the worker, print output.
# ---------------------------------------------------------------------------


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Run a project's training remotely via RunPod Flash")
    parser.add_argument(
        "--project",
        default=os.environ.get("AUTORESEARCH_PROJECT"),
        help="Project under projects/<name>/ (env: AUTORESEARCH_PROJECT)",
    )
    parser.add_argument(
        "--gpu",
        default=os.environ.get("RUNPOD_GPU", DEFAULT_GPU),
        choices=list(GPU_MAP.keys()),
        help=f"GPU type (default: {DEFAULT_GPU})",
    )
    args = parser.parse_args()

    # Re-validate: these should match what was resolved at module load.
    if args.project != _PROJECT or args.gpu != _GPU_NAME:
        print(
            f"ERROR: argparse resolved project='{args.project}' gpu='{args.gpu}' but "
            f"the module-level endpoint was bound to project='{_PROJECT}' gpu='{_GPU_NAME}'. "
            "This should not happen.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.environ.get("RUNPOD_API_KEY"):
        print("ERROR: RUNPOD_API_KEY environment variable is required", file=sys.stderr)
        sys.exit(1)

    train_py = _CFG["train_py_path"].read_text()
    prepare_py = _CFG["prepare_py_path"].read_text()

    print(
        f"Sending to RunPod Flash: project={_PROJECT}, gpu={_GPU_NAME}, "
        f"endpoint={_ENDPOINT_NAME}",
        file=sys.stderr,
    )

    result = await remote_train({
        "train_py": train_py,
        "prepare_py": prepare_py,
        "prepare_timeout_s": _CFG["prepare_timeout_s"],
        "train_timeout_s": _CFG["train_timeout_s"],
    })

    if result["stdout"]:
        print(result["stdout"], end="")
    if result["stderr"]:
        print(result["stderr"], end="", file=sys.stderr)

    sys.exit(result["returncode"])


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
