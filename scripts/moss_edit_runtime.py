"""Shared, lightweight runtime helpers for the editing scripts."""
from __future__ import annotations

import json
from pathlib import Path

import torch

EDIT_CONTRACT = "ref-prefix-only-t0__target-text-original-moss-v1"


def torch_dtype(name):
    allowed = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    if name not in allowed:
        raise ValueError(f"Unsupported dtype {name!r}; choose float32 or bfloat16.")
    return allowed[name]


def read_config(path):
    path = Path(path).expanduser().resolve(strict=True)
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object.")
    # Paths in the JSON are relative to that file, not the launch directory.
    for key in (
        "snapshot_dir", "manifest", "cache_dir", "output_dir",
        "noob_root", "conceptedit_root", "data_index", "noob_pair_index",
    ):
        if config.get(key):
            value = Path(config[key]).expanduser()
            config[key] = str((path.parent / value).resolve() if not value.is_absolute() else value.resolve())
    return config


def load_transformer(snapshot_dir, device="cuda", dtype=torch.bfloat16):
    try:
        from .moss_components import load_moss_module
    except ImportError:
        from moss_components import load_moss_module
    source = Path(snapshot_dir).expanduser().resolve(strict=True)
    model = load_moss_module(source).MOSSImageTransformer.from_pretrained(
        str(source / "transformer"), torch_dtype=dtype,
        local_files_only=True, low_cpu_mem_usage=True,
    )
    return model.requires_grad_(False).to(device)


def model_metadata(base, snapshot_dir):
    return {
        "edit_contract": EDIT_CONTRACT,
        "snapshot_dir": str(Path(snapshot_dir).expanduser().resolve()),
        "model_config": {k: v for k, v in dict(base.config).items() if not k.startswith("_")},
    }
