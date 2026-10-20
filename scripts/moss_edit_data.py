"""Local editing manifests and frozen-encoder caches for MOSS-Image.

These caches contain FLUX.2 latents and frozen Qwen features only. They never
contain DiT activations or KV tensors, which must retain their training graph.
Heavy dependencies are imported inside functions so manifest tools stay light.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EditRecord:
    id: str
    instruction: str
    source_images: tuple[Path, ...]
    target_image: Path
    manifest_line: int


@dataclass
class EncodingComponents:
    prompt_components: Any
    vae: Any
    snapshot_dir: str
    null_prompt_embeds: Any = None
    null_max_sequence_length: int | None = None


def read_manifest(path: str | Path, max_ref_images: int = 4) -> list[EditRecord]:
    """Read JSONL; image paths are relative to the manifest, never the cwd."""
    if max_ref_images < 1:
        raise ValueError("max_ref_images must be positive.")
    path = Path(path).expanduser().resolve(strict=True)
    records, seen = [], set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Each row must be an object.")
                sample_id = row.get("id", f"line-{line_number:08d}")
                if not isinstance(sample_id, str) or not sample_id.strip():
                    raise ValueError("id must be a nonempty string.")
                if sample_id in seen:
                    raise ValueError(f"Duplicate sample id: {sample_id}")
                instruction = row.get("instruction")
                if not isinstance(instruction, str) or not instruction.strip():
                    raise ValueError("instruction must be a nonempty string.")
                if "source_images" in row and "source_image" in row:
                    raise ValueError("Provide source_images or source_image, not both.")
                sources = row.get("source_images")
                if sources is None and "source_image" in row:
                    sources = [row["source_image"]]
                if not isinstance(sources, list) or not 1 <= len(sources) <= max_ref_images:
                    raise ValueError(f"source_images must contain 1..{max_ref_images} paths.")

                def image_path(value):
                    if not isinstance(value, str) or not value:
                        raise ValueError("Image paths must be nonempty strings.")
                    candidate = Path(value).expanduser()
                    if not candidate.is_absolute():
                        candidate = path.parent / candidate
                    candidate = candidate.resolve(strict=True)
                    if not candidate.is_file():
                        raise ValueError(f"Image path is not a file: {candidate}")
                    return candidate

                record = EditRecord(
                    sample_id, instruction, tuple(image_path(v) for v in sources),
                    image_path(row.get("target_image")), line_number,
                )
            except (ValueError, TypeError, OSError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            records.append(record)
            seen.add(sample_id)
    if not records:
        raise ValueError("Manifest contains no editing samples.")
    return records


def resize_image(path: str | Path, max_edge: int = 1024):
    """Return RGB PIL image, capped by max_edge and directly resized to /16.

    Dimensions are independently rounded to the nearest supported size. No
    borders, padding, or crops are introduced. EXIF orientation is honored.
    Small source images are not enlarged except for the minimum 16-pixel side.
    """
    from PIL import Image, ImageOps

    if type(max_edge) is not int or max_edge < 16:
        raise ValueError("max_edge must be an integer >= 16.")
    limit = max_edge // 16 * 16
    with Image.open(path) as opened:
        original = ImageOps.exif_transpose(opened).convert("RGB")
        original.load()
    width, height = original.size
    scale = min(1.0, limit / max(width, height))

    def aligned(value):
        return min(limit, max(16, int(math.floor(value * scale / 16 + 0.5)) * 16))

    image = original.resize((aligned(width), aligned(height)), Image.Resampling.LANCZOS)
    image.info["moss_original_size"] = (width, height)
    return image


def image_to_tensor(image):
    """PIL RGB -> CPU FP32 [1, 3, H, W], normalized to [-1, 1]."""
    import numpy as np
    import torch

    pixels = np.array(image.convert("RGB"), dtype=np.float32, copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).div_(127.5).sub_(1)


def _vae_placement(vae):
    import torch

    parameter = next(vae.parameters(), None)
    if parameter is not None:
        return parameter.device, parameter.dtype
    return vae.bn.running_mean.device, getattr(vae, "dtype", torch.float32)


def _bn_stats(vae, channels: int, device):
    import torch

    mean = vae.bn.running_mean.to(device=device, dtype=torch.float32).reshape(1, -1, 1, 1)
    variance = vae.bn.running_var.to(device=device, dtype=torch.float32).reshape(1, -1, 1, 1)
    eps = float(vae.config.batch_norm_eps)
    if mean.numel() != channels or variance.numel() != channels:
        raise ValueError("Packed latent channels must match FLUX.2 batch-normalization statistics.")
    if not math.isfinite(eps) or eps <= 0 or not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
        raise ValueError("Invalid FLUX.2 batch-normalization statistics.")
    if (variance < 0).any():
        raise ValueError("FLUX.2 running variance must be nonnegative.")
    return mean, (variance + eps).sqrt()


def encode_image(vae, image, *, sample_posterior: bool = False, generator=None):
    """Frozen FLUX.2 encode -> 2x2 channel pack -> BN normalization.

    Input is [B, 3, H, W] in [-1, 1]; result is FP32 [B, 128, H/16, W/16]
    for the official 32-channel VAE. Posterior mode matches Flux2Pipeline.
    """
    import torch

    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("image must be a [B, 3, H, W] tensor.")
    if not image.is_floating_point() or min(image.shape) < 1 or any(s % 16 for s in image.shape[-2:]):
        raise ValueError("Image dimensions must be nonempty multiples of 16 and pixels floating point.")
    if not torch.isfinite(image).all() or image.min() < -1.001 or image.max() > 1.001:
        raise ValueError("Image pixels must be finite and normalized to [-1, 1].")
    device, dtype = _vae_placement(vae)
    with torch.no_grad():
        vae.eval()
        posterior = vae.encode(image.to(device=device, dtype=dtype)).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        if z.ndim != 4 or z.shape[-2] % 2 or z.shape[-1] % 2:
            raise ValueError("VAE latent spatial dimensions must be even for 2x2 packing.")
        batch, channels, height, width = z.shape
        # Official Flux2Pipeline._patchify_latents ordering: c, dy, dx.
        z = z.reshape(batch, channels, height // 2, 2, width // 2, 2)
        z = z.permute(0, 1, 3, 5, 2, 4).reshape(batch, 4 * channels, height // 2, width // 2).float()
        mean, std = _bn_stats(vae, z.shape[1], z.device)
        normalized = (z - mean) / std
        if not torch.isfinite(normalized).all():
            raise ValueError("VAE encoding produced nonfinite latents.")
        return normalized


def decode_latents(vae, latents):
    """Inverse normalized packed latent transform, then frozen VAE decode.

    Returns floating [B, 3, H, W], clamped to [-1, 1] as the official MOSS
    pipeline. Normalization arithmetic is FP32 before conversion to VAE dtype.
    """
    import torch

    if not isinstance(latents, torch.Tensor) or latents.ndim != 4 or min(latents.shape) < 1:
        raise ValueError("latents must be a nonempty [B, C, H, W] tensor.")
    if not latents.is_floating_point() or latents.shape[1] % 4 or not torch.isfinite(latents).all():
        raise ValueError("Latents must be finite floating point, with channels divisible by four.")
    device, dtype = _vae_placement(vae)
    with torch.no_grad():
        vae.eval()
        z = latents.to(device=device, dtype=torch.float32)
        mean, std = _bn_stats(vae, z.shape[1], device)
        z = z * std + mean
        batch, channels, height, width = z.shape
        z = z.reshape(batch, channels // 4, 2, 2, height, width)
        z = z.permute(0, 1, 4, 2, 5, 3).reshape(batch, channels // 4, 2 * height, 2 * width)
        result = vae.decode(z.to(dtype=dtype), return_dict=False)[0]
        if not torch.isfinite(result).all():
            raise ValueError("VAE decoding produced nonfinite images.")
        return result.clamp(-1, 1)


def _adapter_module():
    try:
        from . import moss_components
    except ImportError:
        import moss_components
    return moss_components


def load_encoding_components(snapshot_dir, *, device="cuda", dtype=None, vae_dtype=None):
    """Load the frozen Qwen/tokenizer and FLUX.2 VAE for editing."""
    from importlib.metadata import version

    import torch
    from packaging.version import Version

    if Version(version("diffusers")) < Version("0.38.0"):
        raise RuntimeError(
            "MOSS-Image's asymmetric FLUX.2 VAE requires diffusers>=0.38.0; "
            "older versions do not support decoder_block_out_channels."
        )
    adapter = _adapter_module()
    dtype = torch.bfloat16 if dtype is None else dtype
    vae_dtype = torch.float32 if vae_dtype is None else vae_dtype
    source = Path(snapshot_dir).expanduser().resolve(strict=True)
    prompts = adapter.load_text_components(source, device=device, dtype=dtype)
    module = adapter.load_moss_module(source)
    vae = module.AutoencoderKLFlux2.from_pretrained(
        str(source / "vae"), torch_dtype=vae_dtype, local_files_only=True,
    ).requires_grad_(False).eval().to(device)
    return EncodingComponents(prompts, vae, str(source))


def prepare_sample(record: EditRecord, components: EncodingComponents, *, max_edge=1024,
                   max_sequence_length=4096, sample_posterior=False, generator=None):
    """Encode one sample independently; variable image/text lengths stay unpadded."""

    adapter = _adapter_module()
    if not 1 <= max_sequence_length <= 4096:
        raise ValueError("max_sequence_length must lie in [1, 4096].")
    encoded, sizes, originals = [], [], []
    for path in (*record.source_images, record.target_image):
        image = resize_image(path, max_edge=max_edge)
        encoded.append(encode_image(
            components.vae, image_to_tensor(image), sample_posterior=sample_posterior,
            generator=generator,
        )[0].detach().cpu().contiguous())
        sizes.append(list(image.size))
        originals.append(list(image.info["moss_original_size"]))
    conditions = adapter.encode_prompts(
        components.prompt_components, [record.instruction], negative_prompts=None,
        max_sequence_length=max_sequence_length,
    )
    prompt = conditions.positive.embeds[0, conditions.positive.attention_mask[0].bool()]
    if components.null_prompt_embeds is None or components.null_max_sequence_length != max_sequence_length:
        null = adapter.encode_prompts(
            components.prompt_components, [""], negative_prompts=None,
            max_sequence_length=max_sequence_length,
        ).positive
        components.null_prompt_embeds = null.embeds[0, null.attention_mask[0].bool()].detach().cpu().contiguous()
        components.null_max_sequence_length = max_sequence_length
    sample = {
        "target_latent": encoded[-1], "reference_latents": encoded[:-1],
        "prompt_embeds": prompt.detach().cpu().contiguous(),
        "null_prompt_embeds": components.null_prompt_embeds.clone(),
        "metadata": {
            "format": "moss-edit-cache-v1", "id": record.id,
            "instruction": record.instruction, "source_images": [str(p) for p in record.source_images],
            "target_image": str(record.target_image), "snapshot_dir": components.snapshot_dir,
            "source_sizes": sizes[:-1], "target_size": sizes[-1],
            "source_original_sizes": originals[:-1], "target_original_size": originals[-1],
            "posterior": "sample" if sample_posterior else "mode", "max_edge": max_edge,
            "max_sequence_length": max_sequence_length,
            "latent_space": "flux2-2x2-packed-bn-normalized",
        },
    }
    validate_cached_sample(sample, max_ref_images=len(record.source_images))
    return sample


def validate_cached_sample(sample, *, max_ref_images=4, latent_channels=128, text_dim=None):
    """Reject invalid/nonfinite cache entries before a training worker sees them."""
    import torch

    if not isinstance(sample, dict):
        raise ValueError("Cache must be a plain sample dictionary.")
    required = {"target_latent", "reference_latents", "prompt_embeds", "null_prompt_embeds", "metadata"}
    if set(sample) != required:
        raise ValueError(f"Cache fields must be exactly {sorted(required)}.")
    references = sample["reference_latents"]
    if not isinstance(references, list) or not 1 <= len(references) <= max_ref_images:
        raise ValueError(f"Cache must contain 1..{max_ref_images} reference latents.")
    for latent in [sample["target_latent"], *references]:
        if not isinstance(latent, torch.Tensor) or latent.ndim != 3 or min(latent.shape) < 1:
            raise ValueError("Cached latents must be nonempty [C, H, W] tensors.")
        if latent.shape[0] != latent_channels or not latent.is_floating_point() or not torch.isfinite(latent).all():
            raise ValueError("Cached latent channel count, dtype, or finite-value check failed.")
    prompt, null = sample["prompt_embeds"], sample["null_prompt_embeds"]
    for value in (prompt, null):
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or min(value.shape) < 1:
            raise ValueError("Prompt caches must be nonempty [S, text_dim] tensors.")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError("Prompt caches must contain finite floating-point features.")
        if text_dim is not None and value.shape[1] != text_dim:
            raise ValueError("Cached prompt feature width does not match text_dim.")
    if prompt.shape[1] != null.shape[1]:
        raise ValueError("Conditional and null prompt feature widths differ.")
    if not isinstance(sample["metadata"], dict):
        raise ValueError("metadata must be a JSON-compatible dictionary.")
    try:
        json.dumps(sample["metadata"], allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must contain JSON-compatible finite values.") from error
    return sample


def save_cached_sample(path, sample, *, max_ref_images=4, latent_channels=128, text_dim=None):
    """Atomically publish a validated CPU-only cache; failures leave no partial file."""
    import torch

    validate_cached_sample(sample, max_ref_images=max_ref_images, latent_channels=latent_channels, text_dim=text_dim)
    payload = dict(sample)
    for key in ("target_latent", "prompt_embeds", "null_prompt_embeds"):
        payload[key] = sample[key].detach().cpu().contiguous()
    payload["reference_latents"] = [v.detach().cpu().contiguous() for v in sample["reference_latents"]]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_cached_sample(path, *, max_ref_images=4, latent_channels=128, text_dim=None):
    import torch

    sample = torch.load(path, map_location="cpu", weights_only=True)
    return validate_cached_sample(
        sample, max_ref_images=max_ref_images, latent_channels=latent_channels, text_dim=text_dim,
    )
