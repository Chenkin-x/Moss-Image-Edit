"""Small, unmerged LoRA adapters for the official MOSS transformer.

Load the original transformer first, then inject adapters. Only adapter tensors
are saved; this module never downloads weights or imports PEFT. ``targets`` are
fnmatch globs over full module names. Export directories contain a JSON config
and safetensors, with atomic file replacement and a checksum to detect an
interrupted two-file update. Use a new directory for each training checkpoint.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

_FORMAT = "moss-edit-lora"
_VERSION = 1
_DEFAULT_TARGET = re.compile(r"layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)")
_CONFIG = "adapter_config.json"
_WEIGHTS = "adapter_model.safetensors"


def _validate_options(rank, alpha, dropout, param_dtype):
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("LoRA rank must be a positive integer.")
    if isinstance(alpha, bool) or not math.isfinite(float(alpha)) or float(alpha) <= 0:
        raise ValueError("LoRA alpha must be finite and positive.")
    if not math.isfinite(float(dropout)) or not 0 <= float(dropout) < 1:
        raise ValueError("LoRA dropout must be in [0, 1).")
    if param_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("LoRA param_dtype must be torch.float32 or torch.bfloat16.")


class LoRALinear(nn.Module):
    """Frozen original Linear plus an alpha/rank-scaled A/B update.

    The weight/bias properties intentionally expose the *original* Linear for
    the official MOSS code's input dtype conversions. A/B may retain FP32 while
    the base is BF16. Normal autocast rules still apply to adapter GEMMs.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0,
                 param_dtype: torch.dtype = torch.float32):
        super().__init__()
        _validate_options(rank, alpha, dropout, param_dtype)
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear requires an nn.Linear base.")
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.rank, self.alpha = rank, float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        factory = {"device": base.weight.device, "dtype": param_dtype}
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, **factory))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, **factory))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.train(base.training)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = self.base(x)
        x_lora = self.dropout(x.to(self.lora_A.dtype))
        delta = F.linear(F.linear(x_lora, self.lora_A), self.lora_B)
        return original + (delta * self.scaling).to(original.dtype)


def inject_lora(model: nn.Module, rank: int, alpha: float, dropout: float = 0,
                targets: Sequence[str] | str | None = None,
                param_dtype: torch.dtype = torch.float32) -> list[str]:
    """Freeze all base parameters and replace selected Linear modules in-place.

    No match, a pattern with zero matches, or a non-Linear match is an error.
    Already-injected models are rejected, so accidentally injecting twice
    cannot silently create a different architecture.
    """
    _validate_options(rank, alpha, dropout, param_dtype)
    modules = dict(model.named_modules())
    if any(isinstance(module, LoRALinear) for module in modules.values()):
        raise ValueError("Model already contains LoRA adapters.")
    if targets is None:
        names = [name for name in modules if _DEFAULT_TARGET.fullmatch(name)]
    else:
        patterns = [targets] if isinstance(targets, str) else list(targets)
        if not patterns or any(not isinstance(p, str) or not p for p in patterns):
            raise ValueError("targets must contain nonempty full-name glob patterns.")
        names = []
        for pattern in patterns:
            matches = [name for name in modules if name and fnmatch.fnmatchcase(name, pattern)]
            if not matches:
                raise ValueError(f"LoRA target pattern matched no modules: {pattern!r}")
            names.extend(matches)
    names = sorted(set(names))
    if not names:
        raise ValueError("No LoRA target modules matched the model.")
    for name in names:
        if not isinstance(modules[name], nn.Linear):
            raise TypeError(f"LoRA target {name!r} is not an nn.Linear.")
    replacements = {
        name: LoRALinear(modules[name], rank, alpha, dropout, param_dtype) for name in names
    }
    model.requires_grad_(False)
    for name, module in replacements.items():
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, leaf, module)
    model._moss_adapter_version = 0
    return names


def _adapters(model: nn.Module) -> dict[str, LoRALinear]:
    modules = {name: m for name, m in model.named_modules() if isinstance(m, LoRALinear)}
    if not modules:
        raise ValueError("Model contains no LoRA adapters.")
    return dict(sorted(modules.items()))


def _structure(model: nn.Module) -> dict[str, Any]:
    result = {}
    for name, module in _adapters(model).items():
        if module.lora_A.dtype != module.lora_B.dtype:
            raise ValueError(f"LoRA A/B dtypes disagree for {name}.")
        _validate_options(module.rank, module.alpha, module.dropout.p, module.lora_A.dtype)
        result[name] = {
            "rank": module.rank, "alpha": module.alpha, "dropout": module.dropout.p,
            "in_features": module.in_features, "out_features": module.out_features,
            "param_dtype": str(module.lora_A.dtype).removeprefix("torch."),
        }
    return result


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return independent, contiguous CPU copies of adapter tensors only."""
    return {
        f"{name}.{key}": getattr(module, key).detach().to("cpu").contiguous().clone()
        for name, module in _adapters(model).items()
        for key in ("lora_A", "lora_B")
    }


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


@contextmanager
def _bundle_lock(directory: Path, *, exclusive: bool):
    # flock coordinates this module's readers/writers on Linux training hosts.
    lock = directory / ".adapter.lock"
    if not exclusive and not lock.exists():
        # Exported/read-only bundles may contain just the two public files.
        # The digest and embedded config still reject torn/mixed bundles.
        yield
        return
    import fcntl
    with lock.open("a+b" if exclusive else "rb") as file:
        fcntl.flock(file, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def save_adapter(model: nn.Module, path: str | Path,
                 metadata: Mapping[str, Any] | None = None) -> Path:
    """Save a safetensors bundle, returning its directory.

    Metadata must be JSON serializable; base model identity and edit packing
    conventions are informational. Files are fsynced and atomically replaced; the config is committed
    last. An interrupted overwrite is detected by its weights checksum.
    """
    from safetensors.torch import save_file

    structure = _structure(model)
    config = {
        "format": _FORMAT, "format_version": _VERSION,
        "targets": list(structure), "modules": structure,
        "metadata": dict(metadata or {}),
    }
    serialized = _json(config)
    state = adapter_state_dict(model)
    if any(not bool(torch.isfinite(tensor).all()) for tensor in state.values()):
        raise ValueError("Cannot save non-finite LoRA weights.")
    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    with _bundle_lock(directory, exclusive=True):
        with tempfile.TemporaryDirectory(prefix=".adapter-write-", dir=directory) as tmp:
            tmp = Path(tmp)
            weights, config_path = tmp / _WEIGHTS, tmp / _CONFIG
            save_file(state, str(weights), metadata={"adapter_config": serialized})
            config["weights_sha256"] = _sha256(weights)
            config_path.write_text(_json(config) + "\n", encoding="utf-8")
            for filename in (weights, config_path):
                with filename.open("rb") as file:
                    os.fsync(file.fileno())
            os.replace(weights, directory / _WEIGHTS)
            os.replace(config_path, directory / _CONFIG)
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return directory


def load_adapter(model: nn.Module, path: str | Path) -> dict[str, Any]:
    """Strictly load into an already-injected model; return user metadata.

    Target names, dimensions, rank, alpha, dropout, and adapter dtypes must
    match exactly. Metadata records provenance without restricting base selection.
    Tensor validation precedes any copy.
    Tensor ``copy_`` increments PyTorch parameter versions; a successful load
    also increments ``model._moss_adapter_version`` for external caches.
    """
    from safetensors import safe_open

    directory = Path(path).expanduser()
    structure = _structure(model)
    with _bundle_lock(directory, exclusive=False):
        config = json.loads((directory / _CONFIG).read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("format") != _FORMAT or config.get("format_version") != _VERSION:
            raise ValueError("Unsupported LoRA adapter format/version.")
        if config.get("targets") != list(structure) or config.get("modules") != structure:
            raise ValueError("LoRA structure mismatch: targets/rank/alpha/dropout/dimensions/dtype.")
        metadata = config.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("LoRA metadata must be an object.")
        weights = directory / _WEIGHTS
        if config.get("weights_sha256") != _sha256(weights):
            raise ValueError("LoRA weights checksum mismatch; bundle may be incomplete.")
        with safe_open(str(weights), framework="pt", device="cpu") as file:
            embedded = (file.metadata() or {}).get("adapter_config")
            expected_config = {key: value for key, value in config.items() if key != "weights_sha256"}
            if embedded is None or json.loads(embedded) != expected_config:
                raise ValueError("LoRA safetensors/config metadata mismatch.")
            parameters = {
                f"{name}.{key}": getattr(module, key)
                for name, module in _adapters(model).items() for key in ("lora_A", "lora_B")
            }
            if set(file.keys()) != set(parameters):
                raise ValueError("LoRA weight keys mismatch (missing or unexpected tensors).")
            state = {}
            for name, parameter in parameters.items():
                tensor = file.get_tensor(name)
                if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                    raise ValueError(f"LoRA tensor shape/dtype mismatch for {name}.")
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"Non-finite LoRA tensor: {name}.")
                state[name] = tensor.to(device=parameter.device)
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name])
    model._moss_adapter_version = int(getattr(model, "_moss_adapter_version", 0)) + 1
    return metadata
