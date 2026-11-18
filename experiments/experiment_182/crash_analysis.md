CLASSIFICATION: INFRA

RunPod worker CUDA driver initialization race condition. The traceback shows `torch.cuda.get_device_capability()` (line 19) failed with `RuntimeError: CUDA driver initialization failed, you might not have a CUDA gpu.` The dependency installation log line appears after the crash, confirming the worker executed the script before the CUDA environment was fully initialized. The code changes (chunk_size 256→512, multi-scale decay init [2.0, 3.5, 5.0, 7.0]) are purely constant changes with no new code paths — the crash is unrelated to them. Retrying should succeed.
