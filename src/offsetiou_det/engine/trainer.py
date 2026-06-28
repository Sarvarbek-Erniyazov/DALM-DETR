"""Training loop for OffsetIoU-Det.

Designed for a single 8 GB GPU:
    - AMP (mixed precision) to cut memory and speed up matmuls,
    - gradient accumulation to reach a larger effective batch,
    - gradient clipping (important for stable DETR training),
    - differential learning rate (lower for the pretrained backbone).

The trainer is deliberately framework-light: a plain loop with explicit steps,
so every part is visible and easy to audit.
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

    for it, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=cfg.use_amp):
            pred_logits, pred_boxes = model(images)
            losses = criterion(pred_logits, pred_boxes, targets)
            loss = losses["loss_total"] / cfg.accum_steps

        if not torch.isfinite(loss):
            # NaN-guard: skip this step rather than corrupting the weights.
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()

        # Step every accum_steps mini-batches.
        if (it + 1) % cfg.accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        for k in running:
            running[k] += losses[k].item()
        n_batches += 1

        if cfg.log_every and (it + 1) % cfg.log_every == 0:
            avg = running["loss_total"] / n_batches
            print(f"  epoch {epoch} | iter {it+1}/{len(loader)} | "
                  f"loss {avg:.4f} | {time.time()-t0:.1f}s")

    return {k: v / max(n_batches, 1) for k, v in running.items()}
