"""Small shared utilities: seeding, device resolution, scoped torch runtime settings, logging."""

from __future__ import annotations

import os
import random
import shutil
import warnings
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Seed Python's, NumPy's and torch's (CPU and CUDA) global random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | torch.device | None) -> str:
    """Return the torch device string to use; ``None`` picks the first CUDA device if available."""
    if device is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    device = str(device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={device!r} was requested, but CUDA is not available. Use device='cpu'.")
    return device


def _cpp_compiler_available() -> bool:
    """torch.compile's CPU backend (inductor) generates C++ and needs a compiler at runtime."""
    candidates = [os.environ.get("CXX"), "g++", "c++", "clang++"]
    return any(c and shutil.which(c) for c in candidates)


def resolve_compile(compile_model: bool | str, device: str) -> bool:
    """``"auto"``: compile on CUDA, and on CPU if a C++ compiler is available."""
    if compile_model == "auto":
        if device.startswith("cuda"):
            return True
        return device == "cpu" and _cpp_compiler_available()
    return bool(compile_model)


def compile_module(module: torch.nn.Module, device: str) -> torch.nn.Module:
    """``torch.compile`` the GRANDE network for ``device``.

    CUDA uses ``mode="reduce-overhead"`` (CUDA graphs). CUDA graphs do not apply on CPU; there ``fallback_random`` makes dropout use the eager random number generator. This
    avoids a crash of inductor's vectorized CPU random number code (torch 2.10) and keeps dropout masks
    identical to eager mode.
    """
    if device.startswith("cuda"):
        return torch.compile(module, mode="reduce-overhead", fullgraph=False, dynamic=False)
    return torch.compile(module, fullgraph=False, dynamic=False, options={"fallback_random": True})


#: torch.compile recompiles the network for every new input shape and model configuration (and per
#: batch size of validation/prediction batches). dynamo's default limit of 8 variants per function is
#: quickly reached when several models are fit in one process (e.g. a hyperparameter search), after
#: which training silently falls back to eager mode, 15-35x slower. GRANDE raises the limit while it runs.
_RECOMPILE_LIMIT = 256


@contextmanager
def _dynamo_recompile_limit(limit: int = _RECOMPILE_LIMIT):
    try:
        import torch._dynamo.config as dynamo_config
    except ImportError:  # pragma: no cover
        yield
        return
    patch = {}
    for name in ("recompile_limit", "cache_size_limit"):  # renamed in recent torch versions
        if hasattr(dynamo_config, name):
            if getattr(dynamo_config, name) < limit:
                patch[name] = limit
            break
    if hasattr(dynamo_config, "accumulated_recompile_limit") and dynamo_config.accumulated_recompile_limit < 16 * limit:
        patch["accumulated_recompile_limit"] = 16 * limit
    with dynamo_config.patch(**patch):
        yield


@contextmanager
def torch_runtime(matmul_precision: str | None = None, n_threads: int | None = None):
    """Apply torch settings for the duration of a fit/predict call and restore them afterwards."""
    old_precision = torch.get_float32_matmul_precision()
    old_threads = torch.get_num_threads()
    with _dynamo_recompile_limit(), warnings.catch_warnings():
        # harmless performance note of torch.compile (inductor) about GRANDE's softmax over the estimators
        warnings.filterwarnings("ignore", message=r"\s*Online softmax is disabled")
        # rtdl_num_embeddings: a numeric feature with a single bin (e.g. a binary feature) is expected
        warnings.filterwarnings("ignore", message=r"The \d+-th feature has just two bin edges")
        try:
            if matmul_precision is not None:
                torch.set_float32_matmul_precision(matmul_precision)
            if n_threads is not None:
                torch.set_num_threads(int(n_threads))
            yield
        finally:
            torch.set_float32_matmul_precision(old_precision)
            if torch.get_num_threads() != old_threads:
                torch.set_num_threads(old_threads)


def vprint(verbose: int, level: int, *args) -> None:
    """Print if the verbosity is at least ``level``."""
    if verbose >= level:
        print(*args, flush=True)
