"""Train OffsetIoU-Det on CrowdHuman.

The single most important flag is ``--w_offset``: it is the ablation switch.

    --w_offset 0.0   -> baseline DETR matcher (no location-aware term)
    --w_offset 2.0   -> ours (location-aware matching)

Everything else is held identical between the two runs, so any difference in
MR^-2 / AP / JI is attributable to the matching term alone.

Example:
    python scripts/train.py --w_offset 0.0 --epochs 50 --batch_size 4 --tag baseline
    python scripts/train.py --w_offset 2.0 --epochs 50 --batch_size 4 --tag ours
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from offsetiou_det.data.crowdhuman import CrowdHumanDataset, collate_fn
from offsetiou_det.models.detector import OffsetIoUDet
from offsetiou_det.matching.hungarian import HungarianMatcher
from offsetiou_det.losses.criterion import SetCriterion
from offsetiou_det.engine.trainer import TrainConfig, build_param_groups, train_one_epoch
from offsetiou_det.engine.evaluator import evaluate_model
from offsetiou_det.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train OffsetIoU-Det on CrowdHuman.")
    # Ablation switch
    p.add_argument("--w_offset", type=float, default=0.0,
                   help="weight of the location-aware matching term (0 = baseline)")
    # Data
    p.add_argument("--data_root", default="datasets/crowdhuman")
    p.add_argument("--train_odgt", default="annotation_train.odgt")
    p.add_argument("--val_odgt", default="annotation_val.odgt")
    p.add_argument("--image_size", type=int, default=800)
    # Optimization
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accum_steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--num_queries", type=int, default=300)
    p.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    p.add_argument("--num_workers", type=int, default=2)
    # Bookkeeping
    p.add_argument("--tag", default="run", help="name for checkpoints/logs")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--limit_train", type=int, default=0,
                   help="if >0, use only this many training images (for smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | w_offset={args.w_offset} | tag={args.tag}")

    root = Path(args.data_root)
    img_dir = root / "Images"

    # --- Data ---
    train_ds = CrowdHumanDataset(str(img_dir), str(root / args.train_odgt), args.image_size)
    val_ds = CrowdHumanDataset(str(img_dir), str(root / args.val_odgt), args.image_size)
    if args.limit_train > 0:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(min(args.limit_train, len(train_ds)))))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )
    print(f"train images: {len(train_ds)} | val images: {len(val_ds)}")

    # --- Model / matcher / criterion ---
    model = OffsetIoUDet(num_classes=1, num_queries=args.num_queries,
                         pretrained_backbone=True).to(device)
    matcher = HungarianMatcher(w_offset=args.w_offset)
    criterion = SetCriterion(num_classes=1, matcher=matcher).to(device)

    cfg = TrainConfig(
        epochs=args.epochs, lr=args.lr, lr_backbone=args.lr_backbone,
        accum_steps=args.accum_steps, use_amp=not args.no_amp, device=device,
    )
    optimizer = torch.optim.AdamW(build_param_groups(model, cfg),
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and device == "cuda")

    ckpt_dir = Path(args.out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_mr = float("inf")

    for epoch in range(args.epochs):
        stats = train_one_epoch(model, criterion, train_loader, optimizer,
                                scaler, cfg, epoch)
        scheduler.step()
        print(f"[epoch {epoch}] train loss={stats['loss_total']:.4f} "
              f"(cls={stats['loss_class']:.3f} l1={stats['loss_l1']:.3f} "
              f"giou={stats['loss_giou']:.3f})")

        if (epoch + 1) % args.eval_every == 0:
            metrics = evaluate_model(model, val_loader, device=device,
                                     image_size=args.image_size)
            print(f"[epoch {epoch}] val  MR^-2={metrics['mr']:.4f} "
                  f"AP={metrics['ap']:.4f} JI={metrics['ji']:.4f}")

            if metrics["mr"] < best_mr:
                best_mr = metrics["mr"]
                path = ckpt_dir / f"offsetiou_{args.tag}_best.pth"
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "w_offset": args.w_offset,
                }, path)
                print(f"  saved best checkpoint -> {path} (MR^-2={best_mr:.4f})")

    print(f"Done. Best MR^-2 ({args.tag}) = {best_mr:.4f}")


if __name__ == "__main__":
    main()
