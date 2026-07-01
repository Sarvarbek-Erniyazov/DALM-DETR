"""Generic COCO-format person-detection dataset.

Used for cross-dataset OOD evaluation on CityPersons and WiderPerson, both of
which are exported in standard COCO JSON format (e.g. via Roboflow) to avoid
dataset-specific parsing code -- one loader serves both, keeping the
CrowdHuman -> {CityPersons, WiderPerson} pipeline uniform.

COCO JSON layout (subset of fields we use):
    {
      "images": [{"id": int, "file_name": str, "width": int, "height": int}, ...],
      "annotations": [{"image_id": int, "category_id": int,
                        "bbox": [x, y, w, h], "iscrowd": 0|1}, ...],
      "categories": [{"id": int, "name": str}, ...]
    }

All non-background categories are collapsed to a single "person" class (label
0), matching the CrowdHuman loader's single-class setup, since these datasets
are used purely for zero-shot / cross-dataset evaluation of a person detector,
not for training new classes. Annotations with ``iscrowd=1`` are dropped, as
they are hard-to-localize crowd regions rather than individual boxes.

Output contract (identical to CrowdHumanDataset):
    __getitem__ -> (image_tensor, {"labels", "boxes", "image_id"})
    boxes are normalized (cx, cy, w, h) in [0, 1].
"""

from __future__ import annotations

import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class CocoPersonDataset(Dataset):
    """Person-detection dataset from a COCO-format JSON annotation file.

    Args:
        image_dir:  directory containing the images.
        ann_path:   path to the COCO-format JSON annotation file.
        image_size: square size images are resized to.
        dataset_name: short label (e.g. "citypersons", "widerperson"),
                      stored per-sample for logging / debugging.
    """

    def __init__(
        self,
        image_dir: str,
        ann_path: str,
        image_size: int = 800,
        dataset_name: str = "coco_person",
    ) -> None:
        self.image_dir = image_dir
        self.image_size = image_size
        self.dataset_name = dataset_name

        with open(ann_path, "r") as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.image_ids = list(self.images.keys())

        # Group non-crowd annotations by image_id.
        self.anns_by_image: dict[int, list[dict]] = {img_id: [] for img_id in self.image_ids}
        for ann in coco.get("annotations", []):
            if ann.get("iscrowd", 0) == 1:
                continue
            img_id = ann["image_id"]
            if img_id in self.anns_by_image:
                self.anns_by_image[img_id].append(ann)

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_image(self, file_name: str) -> Image.Image:
        path = os.path.join(self.image_dir, file_name)
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        meta = self.images[img_id]
        img = self._load_image(meta["file_name"])
        orig_w, orig_h = img.size

        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        buf = bytearray(img.tobytes())
        x = torch.frombuffer(buf, dtype=torch.uint8).float() / 255.0
        x = x.view(self.image_size, self.image_size, 3).permute(2, 0, 1)
        mean = torch.tensor(_MEAN).view(3, 1, 1)
        std = torch.tensor(_STD).view(3, 1, 1)
        x = (x - mean) / std

        anns = self.anns_by_image[img_id]
        if anns:
            raw = torch.tensor([a["bbox"] for a in anns], dtype=torch.float32)  # (N,4) xywh
            cx = (raw[:, 0] + raw[:, 2] / 2) / orig_w
            cy = (raw[:, 1] + raw[:, 3] / 2) / orig_h
            w = raw[:, 2] / orig_w
            h = raw[:, 3] / orig_h
            boxes = torch.stack([cx, cy, w, h], dim=1).clamp(0.0, 1.0)
            labels = torch.zeros(len(anns), dtype=torch.long)  # single class: person
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        target = {
            "labels": labels,
            "boxes": boxes,
            "image_id": f"{self.dataset_name}:{meta.get('file_name', img_id)}",
        }
        return x, target


def collate_fn(batch):
    """Stack images; keep targets as a list (variable #boxes per image)."""
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets
