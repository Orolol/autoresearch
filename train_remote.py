# /// script
# requires-python = ">=3.10"
# dependencies = ["runpod-flash"]
# ///
"""
Remote training via RunPod Flash.
Ships train.py and prepare.py to a remote GPU worker, executes them,
and returns stdout/stderr exactly as if run locally.

Usage:
    RUNPOD_API_KEY="xxx" uv run train_remote.py
    RUNPOD_API_KEY="xxx" uv run train_remote.py --gpu A100
    RUNPOD_GPU=A100 uv run train_remote.py
"""

import argparse
import asyncio
import os
import pickle
import sys
from pathlib import Path

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

# ---------------------------------------------------------------------------
# Remote dependencies (installed on the worker via pip)
# ---------------------------------------------------------------------------

REMOTE_DEPS = [
    "torch",
    "kernels",
    "numpy",
    "pyarrow",
    "requests",
    "rustbpe",
    "tiktoken",
]

# ---------------------------------------------------------------------------
# Resolve GPU type from env/CLI at import time (before @Endpoint decoration)
# ---------------------------------------------------------------------------

_gpu_name = os.environ.get("RUNPOD_GPU", DEFAULT_GPU)
_gpu_type = GPU_MAP.get(_gpu_name, GpuType.NVIDIA_H100_80GB_HBM3)

# ---------------------------------------------------------------------------
# Pre-flight: sync existing RunPod endpoints into local pickle cache.
# Prevents "endpoint template names must be unique" errors when the pickle
# is missing an entry for an endpoint that already exists on RunPod.
# ---------------------------------------------------------------------------

def _sync_existing_endpoints():
    """Query RunPod for existing endpoints and add missing ones to the local pickle cache."""
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        return  # Will fail later with a proper error message

    pickle_path = os.path.join(os.path.dirname(__file__) or ".", ".runpod", "resources.pkl")

    # Load existing pickle (or start fresh)
    resources, hashes = {}, {}
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, "rb") as f:
                resources, hashes = pickle.load(f)
        except Exception:
            pass

    # Query RunPod for all endpoints
    try:
        from runpod_flash.core.api.runpod import RunpodGraphQLClient
        from runpod_flash.core.resources.live_serverless import LiveServerless

        async def _query():
            client = RunpodGraphQLClient(api_key=api_key)
            query = 'query { myself { endpoints { id name templateId gpuIds } } }'
            result = await client._execute_graphql(query, {})
            return result['myself']['endpoints']

        endpoints = asyncio.run(_query())
    except Exception:
        return  # Non-fatal: will attempt normal deploy

    # Check each endpoint against our naming convention
    expected_prefix = "autoresearch-train-"
    for ep in endpoints:
        ep_name = ep.get("name", "")
        if not ep_name.startswith(expected_prefix):
            continue
        # Resource key follows the pattern: LiveServerless:{name}
        resource_key = f"LiveServerless:{ep_name}"
        if resource_key in resources:
            continue  # Already in pickle

        # Build a LiveServerless stub from the existing H100 entry or from scratch
        ref = next(iter(resources.values()), None) if resources else None
        try:
            fields = {}
            if ref is not None:
                fields = {field: getattr(ref, field) for field in LiveServerless.model_fields}
            else:
                # Minimal defaults
                fields = {field: info.default for field, info in LiveServerless.model_fields.items()
                          if info.default is not None}
            fields['id'] = ep['id']
            fields['name'] = ep_name
            fields['templateId'] = ep.get('templateId', '')
            fields['gpuIds'] = ep.get('gpuIds', '')
            # Match GPU type from name
            for gpu_cli_name, gpu_type in GPU_MAP.items():
                if gpu_cli_name.lower() in ep_name:
                    fields['gpus'] = [gpu_type]
                    break
            obj = LiveServerless(**fields)
            resources[resource_key] = obj
            hashes[resource_key] = obj.config_hash
            print(f"Synced existing RunPod endpoint: {ep_name} (id={ep['id']})", file=sys.stderr)
        except Exception as e:
            print(f"Warning: could not sync endpoint {ep_name}: {e}", file=sys.stderr)

    # Save updated pickle
    try:
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        with open(pickle_path, "wb") as f:
            pickle.dump((resources, hashes), f)
    except Exception:
        pass

_sync_existing_endpoints()

# ---------------------------------------------------------------------------
# Remote endpoint: receives code as strings, writes files, runs training
# Each GPU type gets its own endpoint name to avoid conflicts.
# ---------------------------------------------------------------------------


@Endpoint(
    name=f"autoresearch-train-{_gpu_name.lower()}",
    gpu=_gpu_type,
    workers=(0, 1),
    dependencies=REMOTE_DEPS,
    execution_timeout_ms=0,
)
async def remote_train(data: dict) -> dict:
    import os
    import subprocess
    import tempfile

    train_py = data["train_py"]
    prepare_py = data["prepare_py"]

    # Write both files into a temp directory
    workdir = tempfile.mkdtemp(prefix="autoresearch_")
    train_path = os.path.join(workdir, "train.py")
    prepare_path = os.path.join(workdir, "prepare.py")

    with open(train_path, "w") as f:
        f.write(train_py)
    with open(prepare_path, "w") as f:
        f.write(prepare_py)

    # Step 1: prepare data (idempotent, skips if already cached in ~/.cache/autoresearch/)
    prep_result = subprocess.run(
        ["python", "prepare.py"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if prep_result.returncode != 0:
        return {
            "stdout": prep_result.stdout,
            "stderr": f"prepare.py failed:\n{prep_result.stderr}",
            "returncode": prep_result.returncode,
        }

    # Step 2: run training
    train_result = subprocess.run(
        ["python", "train.py"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=900,
    )

    return {
        "stdout": train_result.stdout,
        "stderr": train_result.stderr,
        "returncode": train_result.returncode,
    }


# ---------------------------------------------------------------------------
# Local main: read code, ship to remote, print output
# ---------------------------------------------------------------------------


async def async_main():
    parser = argparse.ArgumentParser(description="Run training remotely via RunPod Flash")
    parser.add_argument(
        "--gpu",
        default=os.environ.get("RUNPOD_GPU", DEFAULT_GPU),
        choices=list(GPU_MAP.keys()),
        help=f"GPU type (default: {DEFAULT_GPU})",
    )
    args = parser.parse_args()

    # Validate API key
    if not os.environ.get("RUNPOD_API_KEY"):
        print("ERROR: RUNPOD_API_KEY environment variable is required", file=sys.stderr)
        sys.exit(1)

    # Read local source files
    root = Path(__file__).parent
    train_py = (root / "train.py").read_text()
    prepare_py = (root / "prepare.py").read_text()

    print(f"Sending to RunPod Flash ({args.gpu})...", file=sys.stderr)

    # Call the remote endpoint
    result = await remote_train({"train_py": train_py, "prepare_py": prepare_py})

    # Print stdout exactly as local would
    if result["stdout"]:
        print(result["stdout"], end="")

    # Print stderr to stderr
    if result["stderr"]:
        print(result["stderr"], end="", file=sys.stderr)

    sys.exit(result["returncode"])


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
