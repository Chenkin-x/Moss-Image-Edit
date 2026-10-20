#!/usr/bin/env python3
"""Sample a MOSS editing LoRA with reference-only KV caching and ordinary CFG.

The original base weights are loaded separately from the LoRA. Reference
images remain present in both CFG branches; only the text condition changes.
Qwen and VAE encoding finish before the DiT is loaded onto the sampling device.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.json")
    parser.add_argument("--adapter", required=True, help="Directory containing adapter_config.json and adapter_model.safetensors")
    parser.add_argument("--references", nargs="+", required=True, help="Ordered reference images; normally at most four")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-kv-cache", action="store_true")
    parser.add_argument("--device", default=None, help="Override config device, e.g. cuda:0")
    return parser


def _dtype(name: str):
    import torch
    values = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    if name not in values:
        raise ValueError(f"Unsupported dtype {name!r}; use float32 or bfloat16.")
    return values[name]


def read_adapter_options(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read architecture options before loading the base model; tensors stay unloaded."""
    config = json.loads((Path(path).expanduser() / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("format") != "moss-edit-lora" or config.get("format_version") != 1:
        raise ValueError("Unsupported LoRA adapter format/version.")
    modules, targets = config.get("modules"), config.get("targets")
    if not isinstance(modules, dict) or not modules or targets != sorted(modules):
        raise ValueError("Adapter must contain consistent, nonempty module/target metadata.")
    keys = ("rank", "alpha", "dropout", "param_dtype")
    first = next(iter(modules.values()))
    if not isinstance(first, dict) or not all(key in first for key in keys):
        raise ValueError("Adapter is missing rank/alpha/dropout/param_dtype.")
    shared = {key: first[key] for key in keys}
    if any(not isinstance(value, dict) or any(value.get(key) != shared[key] for key in keys)
           for value in modules.values()):
        raise ValueError("This sampler requires uniform rank/alpha/dropout/dtype across targets.")
    metadata = config.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Adapter metadata must be a JSON object.")
    return {**shared, "targets": targets}, metadata


def validate_arguments(args, config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError("Training config must be a JSON object.")
    if not isinstance(config.get("snapshot_dir"), str) or not config["snapshot_dir"]:
        raise ValueError("Config must specify snapshot_dir.")
    if args.height < 16 or args.width < 16 or args.height % 16 or args.width % 16:
        raise ValueError("height and width must be positive multiples of 16.")
    if args.steps < 1 or not math.isfinite(args.cfg) or args.cfg < 0:
        raise ValueError("steps must be positive and cfg must be finite and nonnegative.")
    if not 0 <= args.seed < 2 ** 63:
        raise ValueError("seed must be between zero and 2**63 - 1.")
    max_references = config.get("max_ref_images", 4)
    if type(max_references) is not int or not 1 <= len(args.references) <= max_references:
        raise ValueError(f"Provide 1..{max_references} reference images.")
    max_length = config.get("max_sequence_length", 4096)
    if type(max_length) is not int or not 1 <= max_length <= 4096:
        raise ValueError("max_sequence_length must be an integer between 1 and 4096.")
    if type(config.get("max_edge", 1024)) is not int or config.get("max_edge", 1024) < 16:
        raise ValueError("max_edge must be an integer >= 16.")
    if Path(args.output).suffix.lower() != ".png":
        raise ValueError("Use a .png output path.")
    for path in args.references:
        if not Path(path).expanduser().is_file():
            raise ValueError(f"Reference image does not exist: {path}")


def sample_latents(editor, scheduler, reference_latents, positive, negative=None, *,
                   height: int, width: int, steps: int = 50, cfg: float = 4,
                   seed: int = 42, device="cuda", latent_channels: int = 128,
                   use_kv_cache: bool = True, progress=None):
    """Euler sampling with one shared reference cache across steps and CFG text.

    Conditions expose ``embeds`` [1,S,D] and ``attention_mask`` [1,S]. Reference
    tensors are [1,C,h,w]. The wrapper receives normalized t; the scheduler
    receives its original 0..1000 timestep. All CFG/Euler arithmetic is FP32.
    """
    import torch

    if min(height, width, steps, latent_channels) < 1 or height % 16 or width % 16:
        raise ValueError("Invalid latent dimensions/steps; image dimensions must be multiples of 16.")
    if not math.isfinite(cfg) or cfg < 0:
        raise ValueError("cfg must be finite and nonnegative.")
    if cfg != 1 and negative is None:
        raise ValueError("CFG requires negative text conditioning.")
    if not reference_latents:
        raise ValueError("Editing requires at least one reference latent.")
    if getattr(scheduler.config, "num_train_timesteps", 1000) != 1000:
        raise ValueError("The edit wrapper requires the official 1000-timestep convention.")
    device = torch.device(device)
    references = [reference.detach().to(device) for reference in reference_latents]
    positive = positive.to(device)
    negative = None if negative is None else negative.to(device)
    generator = torch.Generator("cpu").manual_seed(seed)
    latents = torch.randn(
        (1, latent_channels, height // 16, width // 16),
        generator=generator, device="cpu", dtype=torch.float32,
    ).to(device)
    editor.eval()
    with torch.no_grad():
        cache = editor.prepare_cache(references) if use_kv_cache else None
        scheduler.set_timesteps(steps, device=device)
        total = len(scheduler.timesteps)
        for index, timestep in enumerate(scheduler.timesteps):
            normalized = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(1) / 1000

            def predict(condition):
                return editor(
                    hidden_states=latents, timestep=normalized,
                    encoder_hidden_states=condition.embeds,
                    reference_latents=references,
                    encoder_attention_mask=condition.attention_mask, cache=cache,
                ).float()

            prediction = predict(positive)
            if cfg != 1:
                unconditional = predict(negative)
                prediction = unconditional + cfg * (prediction - unconditional)
            latents = scheduler.step(prediction, timestep, latents, return_dict=False)[0].float()
            if progress is not None:
                progress(index + 1, total)
        if not bool(torch.isfinite(latents).all()):
            raise RuntimeError("Sampling produced non-finite latents.")
    return latents


def _release_cuda(device):
    import torch
    gc.collect()
    if torch.device(device).type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def _save_png(images, output: str | Path) -> Path:
    import torch
    from PIL import Image
    if images.ndim != 4 or images.shape[0] != 1 or images.shape[1] != 3:
        raise ValueError("Expected one decoded RGB image [1,3,H,W].")
    pixels = ((images[0].detach().float().cpu().clamp(-1, 1) + 1) * 127.5)
    pixels = pixels.round().to(torch.uint8).permute(1, 2, 0).numpy()
    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.stem}-", suffix=".png", dir=output.parent)
    os.close(descriptor)
    try:
        Image.fromarray(pixels).save(temporary, format="PNG")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return output


def run(args, config: dict[str, Any]) -> dict[str, Any]:
    """Load frozen encoders, release them, sample with base+adapter, then decode."""
    import torch
    try:
        from . import moss_components as adapter
        from . import moss_edit_data as data
        from . import moss_edit_lora as lora
        from .moss_edit_model import EDIT_CONTRACT, MossEditModel
        from .moss_edit_runtime import load_transformer
    except ImportError:
        import moss_components as adapter
        import moss_edit_data as data
        import moss_edit_lora as lora
        from moss_edit_model import EDIT_CONTRACT, MossEditModel
        from moss_edit_runtime import load_transformer

    options, metadata = read_adapter_options(args.adapter)
    for key in ('text_quantization','dit_quantization'):
        if key in metadata:config.setdefault(key,metadata[key])
    options["param_dtype"] = _dtype(options["param_dtype"])
    snapshot = Path(config["snapshot_dir"]).expanduser().resolve(strict=True)
    device = args.device or config.get("device", "cuda")
    encoder_device = config.get("encoder_device", device)
    # A CLI device override also moves encoding unless the config explicitly
    # requests CPU encoders, preventing an unintended second CUDA device.
    if args.device is not None and encoder_device != "cpu":
        encoder_device = args.device
    encoding = data.load_encoding_components(
        snapshot, device=encoder_device,
        dtype=_dtype(config.get("encoder_dtype", "bfloat16")),
        vae_dtype=_dtype(config.get("vae_dtype", "float32")),
    )
    if config.get('text_quantization') == 'quanto_int8':
        from moss_quanto import quantize_frozen_linears
        quantize_frozen_linears(encoding.prompt_components.text_encoder)
        _release_cuda(encoder_device)
    encoding.prompt_components.offload_text_encoder = True
    with torch.no_grad():
        conditions = adapter.encode_prompts(
            encoding.prompt_components, [args.prompt],
            negative_prompts=[args.negative_prompt] if args.cfg != 1 else None,
            max_sequence_length=config.get("max_sequence_length", 4096), device="cpu",
        )
        reference_latents = [
            data.encode_image(
                encoding.vae,
                data.image_to_tensor(data.resize_image(path, max_edge=config.get("max_edge", 1024))),
                sample_posterior=False,
            ).detach().cpu()
            for path in args.references
        ]
    vae = encoding.vae.to("cpu")
    encoding.prompt_components.text_encoder = None
    del encoding
    _release_cuda(encoder_device)

    print("Reference/text encoding complete; loading base DiT and LoRA.", file=sys.stderr, flush=True)
    model = load_transformer(snapshot, device=device, dtype=_dtype(config.get("dtype", "bfloat16")))
    lora.inject_lora(model, **options)
    lora.load_adapter(model, args.adapter)
    model.requires_grad_(False).eval()
    if config.get('dit_quantization') == 'quanto_int8':
        from moss_quanto import quantize_frozen_linears
        quantize_frozen_linears(model)
        _release_cuda(device)
    packed = config.get('attention_backend') == 'fa4-varlen-mask_mod'
    if packed:
        from moss_edit_varlen import MossVarlenEditModel
        editor = MossVarlenEditModel(model,gradient_checkpointing=False).eval()
    else:
        editor = MossEditModel(model, gradient_checkpointing=False).eval()
    module = adapter.load_moss_module(snapshot)
    scheduler_kwargs = {key: config[source] for key, source in (
        ("shift", "inference_shift"), ("training_shift", "training_shift")
    ) if source in config}
    scheduler = module.MOSSImageScheduler.from_pretrained(
        str(snapshot / "scheduler"), local_files_only=True, **scheduler_kwargs,
    )

    def progress(index, total):
        if index == 1 or index % 5 == 0 or index == total:
            print(f"Sampling {index}/{total}", file=sys.stderr, flush=True)

    latents = sample_latents(
        editor, scheduler, reference_latents, conditions.positive, conditions.negative,
        height=args.height, width=args.width, steps=args.steps, cfg=args.cfg, seed=args.seed,
        device=device, latent_channels=int(model.config.latent_channels),
        use_kv_cache=not (args.no_kv_cache or packed), progress=progress,
    )
    del editor, model, reference_latents, conditions
    _release_cuda(device)
    with torch.no_grad():
        image = data.decode_latents(vae.to(device), latents)
    output = _save_png(image, args.output)
    summary = {
        "image": str(output.resolve()), "adapter": str(Path(args.adapter).expanduser().resolve()),
        "snapshot_dir": str(snapshot), "edit_contract": EDIT_CONTRACT,
        "prompt": args.prompt, "negative_prompt": args.negative_prompt,
        "references": [str(Path(path).expanduser().resolve()) for path in args.references],
        "height": args.height, "width": args.width, "seed": args.seed,
        "requested_steps": args.steps, "actual_steps": len(scheduler.timesteps),
        "cfg": args.cfg, "kv_cache": not (args.no_kv_cache or packed),
        "dit_quantization": config.get('dit_quantization','none'),
        "text_quantization": config.get('text_quantization','none'),
        "inference_shift": float(scheduler.config.shift),
        "training_shift": float(scheduler.config.training_shift),
    }
    output.with_suffix(".png.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if __package__:
            from .moss_edit_runtime import read_config
        else:
            from moss_edit_runtime import read_config
        config = read_config(args.config)
        validate_arguments(args, config)
        result = run(args, config)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
