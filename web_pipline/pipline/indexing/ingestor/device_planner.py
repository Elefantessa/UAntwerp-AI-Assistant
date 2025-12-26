"""
Device Planner
==============

GPU device selection and management for embeddings.
"""

import logging
import os
import subprocess
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore


def _parse_visible_devices() -> Optional[List[int]]:
    """Parse CUDA_VISIBLE_DEVICES environment variable."""
    env_val = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_val is None:
        return None
    try:
        return [int(x.strip()) for x in env_val.split(",") if x.strip()]
    except ValueError:
        return None


def _gpu_free_map_via_torch() -> dict:
    """Get GPU free memory via torch."""
    result = {}
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return result

    for i in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(i)
            result[i] = free
        except Exception:
            pass
    return result


def _gpu_free_map_via_nvsmi() -> dict:
    """Get GPU free memory via nvidia-smi."""
    result = {}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        for line in out.strip().split("\n"):
            parts = line.split(",")
            if len(parts) == 2:
                idx = int(parts[0].strip())
                free_mb = int(parts[1].strip())
                result[idx] = free_mb * 1024 * 1024
    except Exception:
        pass
    return result


def _list_gpus_free_bytes() -> dict:
    """List GPUs with free memory in bytes."""
    result = _gpu_free_map_via_torch()
    if not result:
        result = _gpu_free_map_via_nvsmi()
    return result


class DevicePlanner:
    """
    GPU device planner for optimal device selection.

    Handles device selection with memory requirements and fallback.
    """

    def __init__(self, prefer: Optional[List[int]] = None, min_free_gb: int = 0):
        """
        Initialize device planner.

        Args:
            prefer: Preferred GPU indices
            min_free_gb: Minimum free GPU memory in GB
        """
        vis = _parse_visible_devices()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            default_vis = list(range(torch.cuda.device_count()))
        else:
            default_vis = []

        self.visible: List[int] = prefer if prefer is not None else (vis if vis is not None else default_vis)
        self.min_free = int(min_free_gb) * (1024 ** 3)
        self.tried: List[str] = []

    def pick_best_cuda(self) -> Optional[str]:
        """Pick best available CUDA device."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return None

        free_map = _list_gpus_free_bytes()

        # Filter by minimum memory and visibility
        candidates = []
        for idx in self.visible:
            if idx in free_map and free_map[idx] >= self.min_free:
                candidates.append((idx, free_map[idx]))

        if not candidates:
            return None

        # Sort by free memory (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_idx = candidates[0][0]
        device = f"cuda:{best_idx}"
        self.tried.append(device)
        return device

    def next_best_after(self) -> Optional[str]:
        """Get next best device after current failed."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return None

        free_map = _list_gpus_free_bytes()

        for idx in self.visible:
            device = f"cuda:{idx}"
            if device not in self.tried and idx in free_map:
                if free_map[idx] >= self.min_free:
                    self.tried.append(device)
                    return device

        return None


def resolve_device(device: str, planner: Optional[DevicePlanner] = None) -> str:
    """
    Resolve device string to actual device.

    Args:
        device: Device string ("auto", "cpu", "cuda", "cuda:N")
        planner: Optional device planner

    Returns:
        Resolved device string
    """
    device = device.lower().strip()

    if device == "cpu":
        return "cpu"

    if device.startswith("cuda:"):
        return device

    if device in ("auto", "cuda"):
        if planner:
            cuda_device = planner.pick_best_cuda()
            if cuda_device:
                return cuda_device
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda:0"

    return "cpu"
