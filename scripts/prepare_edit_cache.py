#!/usr/bin/env python3
"""Prepare immutable editing caches, publishing index.json only when complete."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

try:
    from . import moss_edit_data as data
except ImportError:
    import moss_edit_data as data


DEFAULTS = {
    "max_edge": 1024, "max_ref_images": 4, "max_sequence_length": 4096,
    "encoder_device": "cuda", "encoder_dtype": "bfloat16", "vae_dtype": "float32",
}
REQUIRED = {"snapshot_dir", "manifest", "cache_dir"}


def load_config(path):
    """Paths in a configuration resolve relative to that configuration file."""
    path = Path(path).expanduser().resolve(strict=True)
    row = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(row, dict):
        raise ValueError("Configuration must be a JSON object.")
    missing = REQUIRED - row.keys()
    if missing:
        raise ValueError(f"Missing cache preparation configuration fields: {sorted(missing)}")
    # Training, sampling and preparation intentionally share one JSON file.
    # Ignore other stages' keys while validating every preparation field.
    keys = REQUIRED | DEFAULTS.keys()
    config = {**DEFAULTS, **{key: value for key, value in row.items() if key in keys}}
    for key in REQUIRED:
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"{key} must be a nonempty path string.")
        value = Path(config[key]).expanduser()
        config[key] = str((path.parent / value if not value.is_absolute() else value).resolve())
    for key, minimum, maximum in (("max_edge", 16, None), ("max_ref_images", 1, None),
                                  ("max_sequence_length", 1, 4096)):
        value = config[key]
        if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
            raise ValueError(f"Invalid {key}: {value!r}")
    for key in ("encoder_dtype", "vae_dtype"):
        if config[key] not in ("float32", "bfloat16", "float16"):
            raise ValueError(f"{key} must be float32, bfloat16, or float16.")
    if not isinstance(config["encoder_device"], str) or not config["encoder_device"]:
        raise ValueError("encoder_device must be a nonempty device string.")
    if not Path(config["snapshot_dir"]).is_dir():
        raise ValueError("snapshot_dir must be an existing local model directory.")
    if not Path(config["manifest"]).is_file():
        raise ValueError("manifest must be an existing JSONL file.")
    return config


def _atomic_text(path: Path, text: str):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _writer_lock(directory):
    """A second preparation job must not race index publication."""
    import fcntl

    with (directory / ".prepare.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another cache preparation is writing {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def prepare_cache(config, *, overwrite=False):
    """Create a new complete generation; never reuse potentially stale caches.

    --overwrite means recompute into fresh run-specific files, not mutate files
    named by a previously published index. Old runs are retained for readers.
    """
    import torch

    destination = Path(config["cache_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    with _writer_lock(destination):
        existing = [p for p in destination.iterdir() if p.name != ".prepare.lock"]
        if existing and not overwrite:
            raise FileExistsError(
                f"Cache directory is not empty: {destination}. Refusing implicit resume; "
                "use --overwrite to build and publish a new complete cache."
            )
        records = data.read_manifest(config["manifest"], max_ref_images=config["max_ref_images"])
        components = data.load_encoding_components(
            config["snapshot_dir"], device=config["encoder_device"],
            dtype=getattr(torch, config["encoder_dtype"]), vae_dtype=getattr(torch, config["vae_dtype"]),
        )
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10]
        staging = destination / f".staging-{run_id}"
        staging.mkdir()
        relative_run = Path("runs") / run_id
        final_run = destination / relative_run
        samples, failures = [], []
        try:
            for index, record in enumerate(records):
                filename = f"sample_{index:08d}.pt"
                try:
                    source_stats = []
                    for image in (*record.source_images, record.target_image):
                        stat = image.stat()
                        source_stats.append({"path": str(image), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
                    sample = data.prepare_sample(
                        record, components, max_edge=config["max_edge"],
                        max_sequence_length=config["max_sequence_length"],
                    )
                    # Detect files replaced while encoders were reading them.
                    for expected in source_stats:
                        stat = Path(expected["path"]).stat()
                        if (stat.st_size, stat.st_mtime_ns) != (expected["bytes"], expected["mtime_ns"]):
                            raise RuntimeError(f"Source image changed during encoding: {expected['path']}")
                    sample["metadata"]["source_file_stats"] = source_stats
                    data.save_cached_sample(staging / filename, sample, max_ref_images=config["max_ref_images"])
                    samples.append({"file": (relative_run / filename).as_posix(), "id": record.id})
                except Exception as error:
                    failure = {"id": record.id, "manifest_line": record.manifest_line,
                               "error_type": type(error).__name__, "error": str(error)}
                    failures.append(failure)
                    print(json.dumps({"event": "sample_failed", **failure}, ensure_ascii=False), file=sys.stderr, flush=True)
                if (index + 1) % 25 == 0 or index + 1 == len(records):
                    print(json.dumps({"event": "progress", "processed": index + 1,
                                      "succeeded": len(samples), "failed": len(failures)}), flush=True)
            failure_text = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failures)
            _atomic_text(destination / "failures.jsonl", failure_text)
            if not samples:
                raise RuntimeError(f"No samples encoded successfully; see {destination / 'failures.jsonl'}")
            preparation = {key: config[key] for key in sorted(DEFAULTS | {"manifest": None})}
            preparation.update({"posterior": "mode", "run_id": run_id,
                                "succeeded": len(samples), "failed": len(failures)})
            index_payload = {"format": "moss-edit-cache-v1", "samples": samples,
                             "preparation": preparation, "snapshot_dir": config["snapshot_dir"]}
            # Each generation also retains its own provenance and failures.
            _atomic_text(staging / "preparation.json", json.dumps(index_payload, indent=2, ensure_ascii=False) + "\n")
            _atomic_text(staging / "failures.jsonl", failure_text)
            final_run.parent.mkdir(exist_ok=True)
            os.replace(staging, final_run)
            _atomic_text(destination / "index.json", json.dumps(index_payload, indent=2, ensure_ascii=False) + "\n")
            return {"index": str(destination / "index.json"), "run_id": run_id,
                    "succeeded": len(samples), "failed": len(failures)}
        finally:
            if staging.exists():
                shutil.rmtree(staging)
            # Never remove immutable run directories here: interruption can
            # occur immediately after index publication. An unpublished run
            # is harmless; removing one referenced by a reader would not be.


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Cache frozen Qwen features and FLUX.2 latents for MOSS editing LoRA training.",
        epilog="Config paths are relative to the JSON file. Existing caches are never silently resumed.",
    )
    parser.add_argument("--config", required=True, help="JSON configuration file")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute a new cache generation and atomically replace index.json; retain old runs")
    args = parser.parse_args(argv)
    try:
        report = prepare_cache(load_config(args.config), overwrite=args.overwrite)
    except Exception as error:
        print(f"prepare_edit_cache: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"event": "cache_published", **report}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
