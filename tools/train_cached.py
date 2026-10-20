#!/usr/bin/env python3
"""Train reference-conditioned MOSS LoRA from frozen VAE/Qwen feature caches.

Run with python -m tools.train_cached or torchrun --module tools.train_cached.
Each forward is an unpadded, variable-shape
sample. ``samples_per_step`` is per rank; global batch = this * world size.
No detached transformer KV cache is used during training.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from moss_edit.processing import load_cached_sample
from moss_edit.lora import inject_lora, load_adapter, save_adapter
from moss_edit.components import load_transformer, model_metadata, read_config, torch_dtype


class CachedEdits(Dataset):
    def __init__(self, directory, *, max_ref_images=4, latent_channels=128, text_dim=4096):
        self.directory = Path(directory).resolve(strict=True)
        self.index = json.loads((self.directory / "index.json").read_text())
        if self.index.get("format") != "moss-edit-cache-v1" or not self.index.get("samples"):
            raise ValueError("A nonempty prepared editing index.json is required.")
        self.files = []
        for row in self.index["samples"]:
            path = (self.directory / row["file"]).resolve(strict=True)
            if not path.is_relative_to(self.directory) or not path.is_file():
                raise ValueError("Cache index entries must be files inside cache_dir.")
            self.files.append(path)
        self.options = dict(max_ref_images=max_ref_images, latent_channels=latent_channels, text_dim=text_dim)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return load_cached_sample(self.files[index], **self.options)


def single_sample(rows):
    return rows[0]


def sample_stream(dataset, *, rank=0, world_size=1, seed=42, consumed=0, workers=2, pin_memory=False):
    """Epoch shuffle with equal rank lengths; resume skips indices before IO.

    As with DistributedSampler, up to world_size-1 samples can repeat per
    epoch when the dataset size is not divisible by world size.
    """
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=seed, drop_last=False)
    epoch, offset = divmod(consumed, len(sampler))
    while True:
        sampler.set_epoch(epoch)
        indices = list(sampler)[offset:]
        loader = DataLoader(
            dataset, batch_size=1, sampler=indices, num_workers=workers,
            collate_fn=single_sample, pin_memory=pin_memory,
            generator=torch.Generator().manual_seed(seed + epoch),
            **({"prefetch_factor": 2} if workers else {}),
        )
        yield from loader
        epoch, offset = epoch + 1, 0


def flow_loss(model, sample, *, device, dtype, generator, cond_dropout=0.1, shift=1.0):
    """t=0 clean, t=1 noise; only target receives FM noise and supervision."""
    x = sample["target_latent"].unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)
    noise = torch.randn(x.shape, generator=generator, device=device, dtype=torch.float32)
    t = torch.rand((1,), generator=generator, device=device)
    t = shift * t / (1 + (shift - 1) * t)
    drop = bool(torch.rand((), generator=generator, device=device) < cond_dropout)
    context = sample["null_prompt_embeds" if drop else "prompt_embeds"]
    context = context.unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True)
    refs = [v.unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True)
            for v in sample["reference_latents"]]
    prediction = model(((1 - t[:, None, None, None]) * x + t[:, None, None, None] * noise).to(dtype),
                       t, context, refs)
    return (prediction.float() - (noise - x)).square().mean()


def _rng_state(generator, device):
    return {
        "torch_cpu": torch.get_rng_state(), "noise": generator.get_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _restore_rng(state, generator, device):
    torch.set_rng_state(state["torch_cpu"])
    generator.set_state(state["noise"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def save_checkpoint(base, optimizer, directory, *, step, consumed, config, metadata,
                    generator, device, rank=0, world_size=1, dataset_index=None):
    rng = _rng_state(generator, device)
    states = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(states, rng)
    else:
        states[0] = rng
    if rank == 0:
        destination = Path(directory) / f"checkpoint-{step:08d}"
        if destination.exists():
            raise FileExistsError(f"Checkpoint already exists: {destination}")
        temporary = destination.with_name(f".{destination.name}-{uuid.uuid4().hex}.tmp")
        try:
            save_adapter(base, temporary, metadata)
            torch.save({
                "format": "moss-edit-training-v1", "step": step, "consumed_per_rank": consumed,
                "world_size": world_size, "config": config, "optimizer": optimizer.state_dict(),
                "rng_by_rank": states, "dataset_index": dataset_index,
            }, temporary / "training_state.pt")
            os.rename(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        print(f"Saved {destination}", flush=True)
    if world_size > 1:
        dist.barrier()


def train(config, *, resume=None):
    # Imported lazily so --help works before transformer dependencies are installed.
    from moss_edit.model import MossEditModel

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment.")
        device = torch.device("cuda", local_rank if world > 1 else (device.index or 0))
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        seed = int(config.get("seed", 42))
        torch.manual_seed(seed)
        dtype = torch_dtype(config.get("dtype", "bfloat16"))
        max_steps = int(config.get("max_steps", 10000))
        samples_per_step = int(config.get("samples_per_step", 4))
        save_every = int(config.get("save_every", 500))
        cond_dropout = float(config.get("cond_dropout", 0.1))
        shift = float(config.get("training_shift", 1.0))
        lr = float(config.get("learning_rate", 1e-4))
        max_norm = float(config.get("max_grad_norm", 1.0))
        if min(max_steps, samples_per_step, save_every) < 1 or not 0 <= cond_dropout <= 1:
            raise ValueError("Steps/batch/save interval must be positive; cond_dropout in [0,1].")
        if not math.isfinite(shift) or shift <= 0 or not math.isfinite(lr) or lr <= 0:
            raise ValueError("training_shift and learning_rate must be finite positive numbers.")
        if not math.isfinite(max_norm) or max_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive.")
        output = Path(config["output_dir"]).resolve()
        if output.exists() and any(output.iterdir()) and not resume:
            raise FileExistsError(f"Output directory is nonempty; choose a new one or --resume: {output}")
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
        if world > 1:
            dist.barrier()
        base = load_transformer(config["snapshot_dir"], device=device, dtype=dtype)
        inject_lora(base, rank=int(config.get("lora_rank", 32)), alpha=float(config.get("lora_alpha", 32)),
                    dropout=float(config.get("lora_dropout", 0)), targets=config.get("lora_targets"),
                    param_dtype=torch_dtype(config.get("lora_dtype", "float32")))
        metadata = model_metadata(base, config["snapshot_dir"])
        data = CachedEdits(config["cache_dir"], max_ref_images=int(config.get("max_ref_images", 4)),
                           latent_channels=base.config.latent_channels, text_dim=base.config.text_dim)
        model = MossEditModel(base, gradient_checkpointing=bool(config.get("gradient_checkpointing", True)))
        model.train()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=lr, betas=(0.9, 0.999),
                                      weight_decay=float(config.get("weight_decay", 0.01)))
        generator = torch.Generator(device=device).manual_seed(seed + 104729 * rank)
        torch.manual_seed(seed + rank)
        step, consumed = 0, 0
        if resume:
            state = torch.load(Path(resume) / "training_state.pt", map_location="cpu", weights_only=True)
            if state.get("format") != "moss-edit-training-v1" or state["world_size"] != world:
                raise ValueError("Resume requires matching format and world size.")
            if state.get("dataset_index") != data.index:
                raise ValueError("Cache index changed; exact training resume cannot use different samples.")
            # Base selection is independent of adapter provenance.
            resume_mutable = {"snapshot_dir", "max_steps", "save_every", "num_workers", "output_dir", "wandb"}
            old, new = state["config"], config
            if {k: v for k, v in old.items() if k not in resume_mutable} != {
                k: v for k, v in new.items() if k not in resume_mutable
            }:
                raise ValueError("Resume configuration changed beyond base/steps/save interval/workers/output/wandb.")
            load_adapter(base, resume)
            optimizer.load_state_dict(state["optimizer"])
            step, consumed = state["step"], state["consumed_per_rank"]
            _restore_rng(state["rng_by_rank"][rank], generator, device)
        if world > 1:
            model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                        broadcast_buffers=False, find_unused_parameters=False)
        stream = sample_stream(data, rank=rank, world_size=world, seed=seed, consumed=consumed,
                               workers=int(config.get("num_workers", 2)), pin_memory=device.type == "cuda")
        wandb_run = None
        if rank == 0:
            if resume:
                newer = [p for p in output.glob("checkpoint-*")
                         if p.name.removeprefix("checkpoint-").isdigit()
                         and int(p.name.removeprefix("checkpoint-")) > step]
                if newer:
                    raise ValueError("This output contains newer checkpoints; resume an older one into a new output_dir.")
                metrics_path = output / "metrics.jsonl"
                if metrics_path.exists():
                    lines = metrics_path.read_text().splitlines()
                    kept = []
                    for line in lines:
                        try:
                            if json.loads(line)["step"] <= step:
                                kept.append(line)
                        except (ValueError, KeyError):
                            pass  # A crash can leave an incomplete last line.
                    if kept != lines:
                        metrics_path.rename(output / f"metrics-before-resume-{uuid.uuid4().hex[:8]}.jsonl")
                        metrics_path.write_text("\n".join(kept) + ("\n" if kept else ""))
            (output / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
            print(json.dumps({"trainable_parameters": sum(p.numel() for p in parameters),
                              "cached_samples": len(data), "global_batch": samples_per_step * world,
                              "start_step": step, "max_steps": max_steps}), flush=True)
            if config.get("wandb", {}).get("enabled"):
                import wandb
                options = config["wandb"]
                wandb_run = wandb.init(project=options.get("project", "moss-edit-lora"),
                                       name=options.get("name"), config=config, dir=str(output))
        last_saved = step if resume else -1
        while step < max_steps:
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=device)
            for micro in range(samples_per_step):
                sample = next(stream)
                sync = model.no_sync() if world > 1 and micro + 1 < samples_per_step else nullcontext()
                with sync:
                    with torch.autocast(device_type=device.type, dtype=dtype,
                                        enabled=device.type == "cuda" and dtype == torch.bfloat16):
                        loss = flow_loss(model, sample, device=device, dtype=dtype, generator=generator,
                                         cond_dropout=cond_dropout, shift=shift)
                    (loss / samples_per_step).backward()
                loss_sum += loss.detach()
                consumed += 1
            norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
            if not torch.isfinite(loss_sum):
                raise FloatingPointError(f"Nonfinite training loss before optimizer step {step + 1}.")
            optimizer.step()
            step += 1
            if world > 1:
                dist.all_reduce(loss_sum)
            metrics = {"step": step, "train/loss": float(loss_sum / (samples_per_step * world)),
                       "train/grad_norm": float(norm), "train/step_seconds": time.monotonic() - started,
                       "train/images_seen": consumed * world}
            if rank == 0:
                print(json.dumps(metrics), flush=True)
                with (output / "metrics.jsonl").open("a") as file:
                    file.write(json.dumps(metrics) + "\n")
                if wandb_run:
                    wandb_run.log(metrics, step=step)
            if step % save_every == 0 or step == max_steps:
                save_checkpoint(base, optimizer, output, step=step, consumed=consumed, config=config,
                                metadata=metadata, generator=generator, device=device, rank=rank,
                                world_size=world, dataset_index=data.index)
                last_saved = step
        if wandb_run:
            wandb_run.finish()
        return {"step": step, "consumed_per_rank": consumed, "last_saved_step": last_saved}
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON config; paths resolve relative to this file.")
    parser.add_argument("--resume", help="A complete checkpoint directory (adapter + training_state.pt).")
    parser.add_argument("--max-steps", type=int, help="Override the total optimizer-step limit.")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()
