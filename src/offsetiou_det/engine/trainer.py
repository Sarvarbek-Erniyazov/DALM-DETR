"""Training loop for OffsetIoU-Det.

Designed for a single 8 GB GPU:
    - AMP (mixed precision) to cut memory and speed up matmuls,
    - gradient accumulation to reach a larger effective batch,
    - gradient clipping (important for stable DETR training),
    - differential learning rate (lower for the pretrained backbone).

The trainer is deliberately framework-light: a plain loop with explicit steps,
so every part is visible and easy to audit.

PROFILING: set cfg.profile=True to print a periodic breakdown of where time is
spent per step: data loading (waiting on the DataLoader) vs. compute (forward +
matcher + backward). This isolates whether a slowdown comes from I/O/CPU
(data loading) or GPU-side compute (model + matching).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    clip_max_norm: float = 0.1
    accum_steps: int = 2           # effective batch = batch_size * accum_steps
    use_amp: bool = True
    log_every: int = 50
    device: str = "cuda"
    profile: bool = False          # log data-load vs compute time breakdown


def build_param_groups(model: nn.Module, cfg: TrainConfig):
    """Two param groups: backbone (low LR) and everything else (base LR)."""
    backbone_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(p)
        else:
            other_params.append(p)
    return [
        {"params": backbone_params, "lr": cfg.lr_backbone},
        {"params": other_params, "lr": cfg.lr},
    ]


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: TrainConfig,
    epoch: int,
) -> dict[str, float]:
    """Run one training epoch. Returns averaged loss components."""
    model.train()
    criterion.train()
    device = cfg.device

    running = {"loss_total": 0.0, "loss_class": 0.0, "loss_l1": 0.0, "loss_giou": 0.0}
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()

    # Profiling accumulators.
    data_time_sum = 0.0
    compute_time_sum = 0.0
    matcher_time_sum = 0.0
    max_targets_seen = 0

    data_iter = iter(loader)
    it = 0
    while True:
        t_data_start = time.time()
        try:
            images, targets = next(data_iter)
        except StopIteration:
            break
        if cfg.profile and device == "cuda":
            torch.cuda.synchronize()
        data_time_sum += time.time() - t_data_start

        t_compute_start = time.time()
        images = images.to(device, non_blocking=True)

        if cfg.profile:
            max_targets_seen = max(max_targets_seen, max((len(t["labels"]) for t in targets), default=0))

        with torch.autocast(device_type="cuda", enabled=cfg.use_amp):
            pred_logits, pred_boxes = model(images)

            if cfg.profile and device == "cuda":
                torch.cuda.synchronize()
            t_matcher_start = time.time()

            losses = criterion(pred_logits, pred_boxes, targets)

            if cfg.profile and device == "cuda":
                torch.cuda.synchronize()
            matcher_time_sum += time.time() - t_matcher_start

            loss = losses["loss_total"] / cfg.accum_steps

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            it += 1
            continue

        scaler.scale(loss).backward()

        if (it + 1) % cfg.accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if cfg.profile and device == "cuda":
            torch.cuda.synchronize()
        compute_time_sum += time.time() - t_compute_start

        for k in running:
            running[k] += losses[k].item()
        n_batches += 1
        it += 1

        if cfg.log_every and it % cfg.log_every == 0:
            avg = running["loss_total"] / n_batches
            msg = (f"  epoch {epoch} | iter {it} | loss {avg:.4f} | "
                   f"{time.time()-t0:.1f}s")
            if cfg.profile:
                msg += (f" | data={data_time_sum:.1f}s compute={compute_time_sum:.1f}s "
                        f"matcher={matcher_time_sum:.1f}s max_gt={max_targets_seen}")
            print(msg)

    return {k: v / max(n_batches, 1) for k, v in running.items()}
