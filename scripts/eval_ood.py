"""Cross-dataset OOD evaluation: CrowdHuman-trained model -> CityPersons / WiderPerson.

No fine-tuning. Loads a checkpoint, runs evaluate_model (same metrics as val:
MR^-2, AP, JI), prints one table row.

Usage:
  python scripts/eval_ood_v2.py --checkpoint outputs/checkpoints/offsetiou_baseline_v2_best.pth \
      --dataset widerperson --data_root data/WiderPerson --image_size 640
  python scripts/eval_ood_v2.py --checkpoint ... --dataset citypersons \
      --data_root data/CityPersons --ann_file data/CityPersons/val_coco.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offsetiou_det.engine.evaluator import evaluate_model  # noqa: E402

# ImageNet stats — training transform bilan BIR XIL bo'lishi shart (QADAM 0 da tekshiramiz)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class WiderPersonDataset(Dataset):
    """WiderPerson val split.

    Layout:
      data_root/Images/*.jpg
      data_root/Annotations/<name>.jpg.txt   (line1: count; then: cls x1 y1 x2 y2)
      data_root/val.txt                      (image ids, one per line)
    Keeps classes 1-3 (pedestrians, riders, partially-visible); skips 4 (crowd), 5 (ignore).
    """

    def __init__(self, root: str, image_size: int):
        self.root = Path(root)
        self.image_size = image_size
        ids = (self.root / "val.txt").read_text().split()
        self.items = [i.strip() for i in ids if i.strip()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        name = self.items[idx]
        img = Image.open(self.root / "Images" / f"{name}.jpg").convert("RGB")
        W, H = img.size

        boxes = []
        ann = self.root / "Annotations" / f"{name}.jpg.txt"
        lines = ann.read_text().splitlines()
        for line in lines[1:]:
            p = line.split()
            if len(p) < 5:
                continue
            cls, x1, y1, x2, y2 = int(p[0]), *map(float, p[1:5])
            if cls > 3:
                continue
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            w, h = (x2 - x1) / W, (y2 - y1) / H
            if w <= 0 or h <= 0:
                continue
            boxes.append([cx, cy, w, h])

        img = TF.normalize(
            TF.to_tensor(TF.resize(img, [self.image_size, self.image_size])),
            MEAN, STD,
        )
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        return img, {"boxes": boxes, "labels": torch.zeros(len(boxes), dtype=torch.long)}


class CocoPersonDataset(Dataset):
    """COCO-format json (CityPersons converted). bbox = [x, y, w, h] pixels."""

    def __init__(self, root: str, ann_file: str, image_size: int):
        self.root = Path(root)
        self.image_size = image_size
        coco = json.loads(Path(ann_file).read_text())
        self.images = {im["id"]: im for im in coco["images"]}
        self.anns: dict[int, list] = {}
        for a in coco["annotations"]:
            if a.get("iscrowd", 0) or a.get("ignore", 0):
                continue
            self.anns.setdefault(a["image_id"], []).append(a["bbox"])
        self.ids = sorted(self.images.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        info = self.images[self.ids[idx]]
        img = Image.open(self.root / info["file_name"]).convert("RGB")
        W, H = img.size
        boxes = []
        for (x, y, w, h) in self.anns.get(self.ids[idx], []):
            if w <= 0 or h <= 0:
                continue
            boxes.append([(x + w / 2) / W, (y + h / 2) / H, w / W, h / H])
        img = TF.normalize(
            TF.to_tensor(TF.resize(img, [self.image_size, self.image_size])),
            MEAN, STD,
        )
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        return img, {"boxes": boxes, "labels": torch.zeros(len(boxes), dtype=torch.long)}


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    tgts = [b[1] for b in batch]
    return imgs, tgts


def build_model_from_checkpoint(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    # train.py bilan ayni moslik:
    from offsetiou_det.models.detector import OffsetIoUDet
    model = OffsetIoUDet(num_classes=1, num_queries=300, pretrained_backbone=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", choices=["widerperson", "citypersons"], required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ann_file", default=None, help="COCO json (citypersons uchun)")
    ap.add_argument("--image_size", type=int, default=640)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test: faqat N ta rasm")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ikkala OOD dataset ham Roboflow COCO formatida (leakage audit bilan bir xil nusxa)
    assert args.ann_file, "--ann_file kerak (COCO json)"
    ds = CocoPersonDataset(args.data_root, args.ann_file, args.image_size)

    if args.limit:
        ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, collate_fn=collate)
    model = build_model_from_checkpoint(args.checkpoint, device)

    metrics = evaluate_model(model, loader, device=device, image_size=args.image_size)
    print(f"\n[OOD:{args.dataset}] ckpt={Path(args.checkpoint).name} "
          f"images={len(ds)} | MR^-2={metrics['mr']:.4f} "
          f"AP={metrics['ap']:.4f} JI={metrics['ji']:.4f}")


if __name__ == "__main__":
    main()
