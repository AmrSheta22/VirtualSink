"""Report the Slurm-visible GPU and verify a CUDA forward/backward pass."""
import json
import ctypes
import os
import platform
import socket

import numpy
import torch

if not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("Submit this diagnostic through Slurm")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in this allocation")
report = {
    "job_id": os.environ["SLURM_JOB_ID"],
    "host": socket.gethostname(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "numpy": numpy.__version__,
    "cuda_runtime": torch.version.cuda,
    "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "gpus": [],
}
driver = ctypes.CDLL("libcuda.so.1")
driver_version = ctypes.c_int()
if driver.cuDriverGetVersion(ctypes.byref(driver_version)) == 0:
    report["cuda_driver_api_version"] = driver_version.value
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    report["gpus"].append({
        "index": i, "name": p.name, "memory_gib": p.total_memory / 1024**3,
        "compute_capability": f"{p.major}.{p.minor}",
        "multiprocessors": p.multi_processor_count,
    })
x = torch.randn(64, 64, device="cuda", requires_grad=True)
(x @ x).square().mean().backward()
torch.cuda.synchronize()
report["forward_backward"] = "passed"
print(json.dumps(report, indent=2), flush=True)
