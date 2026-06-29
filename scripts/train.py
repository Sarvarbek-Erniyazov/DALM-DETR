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
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

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
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_max_norm", type=float, default=0.1)
    p.add_argument("--num_queries", type=int, default=300)
    p.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--warmup_epochs", type=int, default=1,
                   help="linear LR warmup before cosine annealing")
    # Early stopping
    p.add_argument("--patience", type=int, default=10,
                   help="stop if val MR^-2 does not improve for this many evals")
    p.add_argument("--min_delta", type=float, default=1e-4,
                   help="minimum MR^-2 improvement to reset patience")
    # Bookkeeping
    p.add_argument("--tag", default="run", help="name for checkpoints/logs")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--limit_train", type=int, default=0,
                   help="if >0, use only this many training images (smoke tests)")
    p.add_argument("--limit_val", type=int, default=0,
                   help="if >0, evaluate on only this many val images (smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default="",
                   help="path to a checkpoint to resume model/optimizer/epoch from")
    return p.parse_args()


def build_scheduler(optimizer, args):
    """Linear warmup -> cosine annealing."""
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=max(args.warmup_epochs, 1)
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - args.warmup_epochs, 1)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[max(args.warmup_epochs, 1)]
    )


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
        train_ds = Subset(train_ds, list(range(min(args.limit_train, len(train_ds)))))
    if args.limit_val > 0:
        val_ds = Subset(val_ds, list(range(min(args.limit_val, len(val_ds)))))

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
        weight_decay=args.weight_decay, clip_max_norm=args.clip_max_norm,
        accum_steps=args.accum_steps, use_amp=not args.no_amp, device=device,
    )
    optimizer = torch.optim.AdamW(build_param_groups(model, cfg),
                                  weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and device == "cuda")

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_mr = float("inf")
    epochs_no_improve = 0
    history = []
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_mr = ckpt.get("best_mr", best_mr)
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        stats = train_one_epoch(model, criterion, train_loader, optimizer,
                                scaler, cfg, epoch)
        scheduler.step()
        cur_lr = optimizer.param_groups[-1]["lr"]
        print(f"[epoch {epoch}] train loss={stats['loss_total']:.4f} "
              f"(cls={stats['loss_class']:.3f} l1={stats['loss_l1']:.3f} "
              f"giou={stats['loss_giou']:.3f}) lr={cur_lr:.2e} "
              f"{time.time()-t0:.1f}s")

        record = {"epoch": epoch, "train_loss": stats["loss_total"], "lr": cur_lr}

        if (epoch + 1) % args.eval_every == 0:
            metrics = evaluate_model(model, val_loader, device=device,
                                     image_size=args.image_size)
            print(f"[epoch {epoch}] val  MR^-2={metrics['mr']:.4f} "
                  f"AP={metrics['ap']:.4f} JI={metrics['ji']:.4f}")
            record.update({f"val_{k}": v for k, v in metrics.items()})

            improved = metrics["mr"] < best_mr - args.min_delta
            if improved:
                best_mr = metrics["mr"]
                epochs_no_improve = 0
                path = ckpt_dir / f"offsetiou_{args.tag}_best.pth"
                torch.save({
                    "model": model.state_dict(), "epoch": epoch,
                    "metrics": metrics, "w_offset": args.w_offset,
                }, path)
                print(f"  saved best -> {path} (MR^-2={best_mr:.4f})")
            else:
                epochs_no_improve += 1
                print(f"  no improvement ({epochs_no_improve}/{args.patience})")

        history.append(record)
        with open(log_dir / f"history_{args.tag}.json", "w") as f:
            json.dump(history, f, indent=2)

        # Always keep a "last" checkpoint so a long run can resume / be inspected.
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "epoch": epoch, "best_mr": best_mr,
            "w_offset": args.w_offset,
        }, ckpt_dir / f"offsetiou_{args.tag}_last.pth")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} "
                  f"(no MR^-2 improvement for {args.patience} evals).")
            break

    print(f"Done. Best MR^-2 ({args.tag}) = {best_mr:.4f}")


if __name__ == "__main__":
    main()
