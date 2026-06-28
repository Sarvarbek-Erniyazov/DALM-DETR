"""CrowdHuman dataset for OffsetIoU-Det.

CrowdHuman annotations come in ``.odgt`` files: one JSON object per line.
Each object has:
    {"ID": image_name, "gtboxes": [{"tag": "person"|"mask",
                                    "fbox": [x, y, w, h], ...,
                                    "extra": {"ignore": 0|1}}, ...]}

We use the full-body box (``fbox``) of every ``person`` whose ``ignore`` flag
is not set. Boxes are returned in normalized (cx, cy, w, h) format to match the
model and matcher. Images are resized to a fixed square so they batch cleanly
on an 8 GB GPU.
"""

from __future__ import annotations

import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

# ImageNet normalization (matches the pretrained ResNet backbone).
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _parse_odgt(odgt_path: str) -> list[dict]:
    """Read an .odgt file into a list of per-image annotation records."""
    with open(odgt_path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _extract_person_fboxes(record: dict) -> list[list[float]]:
    """Return full-body boxes [x, y, w, h] for non-ignored persons."""
    boxes = []
    for gt in record.get("gtboxes", []):
        if gt.get("tag") != "person":
            continue
        extra = gt.get("extra", {})
        if extra.get("ignore", 0) == 1:
            continue
        x, y, w, h = gt["fbox"]
        if w <= 0 or h <= 0:
            continue
        boxes.append([float(x), float(y), float(w), float(h)])
    return boxes


class CrowdHumanDataset(Dataset):
    """CrowdHuman detection dataset.

    Args:
        image_dir: directory containing the .jpg images.
        odgt_path: path to annotation_train.odgt / annotation_val.odgt.
        image_size: square size images are resized to.
        transforms: optional callable(image_tensor, target) -> (image, target)
                    applied after the base resize/normalize.
    """

    def __init__(
        self,
        image_dir: str,
        odgt_path: str,
        image_size: int = 800,
        transforms=None,
    ) -> None:
        self.image_dir = image_dir
        self.image_size = image_size
        self.transforms = transforms
        self.records = _parse_odgt(odgt_path)

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, image_id: str) -> Image.Image:
        for ext in (".jpg", ".png", ""):
            path = os.path.join(self.image_dir, image_id + ext)
            if os.path.exists(path):
                return Image.open(path).convert("RGB")
        raise FileNotFoundError(f"image not found for ID={image_id} in {self.image_dir}")

    def __getitem__(self, idx: int):
        record = self.records[idx]
        img = self._load_image(record["ID"])
        orig_w, orig_h = img.size

        # Resize to a fixed square.
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Image -> normalized CHW tensor (no torchvision.transforms dependency).
        buf = bytearray(img.tobytes())
        x = torch.frombuffer(buf, dtype=torch.uint8).float() / 255.0
        x = x.view(self.image_size, self.image_size, 3).permute(2, 0, 1)
        mean = torch.tensor(_MEAN).view(3, 1, 1)
        std = torch.tensor(_STD).view(3, 1, 1)
        x = (x - mean) / std

        # Boxes: [x, y, w, h] in original pixels -> normalized (cx, cy, w, h).
        raw = _extract_person_fboxes(record)
        if raw:
            b = torch.tensor(raw, dtype=torch.float32)
            cx = (b[:, 0] + b[:, 2] / 2) / orig_w
            cy = (b[:, 1] + b[:, 3] / 2) / orig_h
            w = b[:, 2] / orig_w
            h = b[:, 3] / orig_h
            boxes = torch.stack([cx, cy, w, h], dim=1).clamp(0.0, 1.0)
            labels = torch.zeros(len(raw), dtype=torch.long)  # single class: person
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        target = {"labels": labels, "boxes": boxes, "image_id": record["ID"]}

        if self.transforms is not None:
            x, target = self.transforms(x, target)
        return x, target


def collate_fn(batch):
    """Stack images; keep targets as a list (variable #boxes per image)."""
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets
