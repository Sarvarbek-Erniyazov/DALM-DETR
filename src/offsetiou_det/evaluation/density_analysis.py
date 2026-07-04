"""Density-stratified evaluation.

The central claim of this project is that a location-aware matching term helps
*where crowding is worst*. A single dataset-level number cannot test that
claim: a method could win on sparse images and lose in dense clusters, or vice
versa, and the average would hide it. This module therefore stratifies the
evaluation set by ground-truth person count per image and reports MR^-2 / AP /
JI separately per stratum.

Strata (person count per image):
    sparse:  1  <= n < 10
    medium:  10 <= n < 25
    dense:   n >= 25

The expected signature of the project hypothesis is a monotone pattern:
roughly equal performance on sparse (where IoU-based matching is already
unambiguous) and a widening gap in favor of the offset variants on dense.

Strata are formed by image, not by box, so each stratum is a self-contained
mini-dataset and the metrics remain well-defined (FPPI in MR^-2 is per-image).
"""

from __future__ import annotations

import numpy as np

from .crowd_metrics import evaluate

# (name, lower bound inclusive, upper bound exclusive)
DEFAULT_STRATA = (
    ("sparse", 1, 10),
    ("medium", 10, 25),
    ("dense", 25, 10**9),
)


def stratify_by_density(
    predictions: list[dict],
    ground_truths: list[np.ndarray],
    strata=DEFAULT_STRATA,
) -> dict[str, dict]:
    """Split (predictions, ground_truths) into density strata by GT count."""
    out = {name: {"predictions": [], "ground_truths": []} for name, _, _ in strata}
    for pred, gt in zip(predictions, ground_truths):
        n = len(gt)
        for name, lo, hi in strata:
            if lo <= n < hi:
                out[name]["predictions"].append(pred)
                out[name]["ground_truths"].append(gt)
                break
    for name in out:
        out[name]["num_images"] = len(out[name]["ground_truths"])
        out[name]["num_gt"] = int(sum(len(g) for g in out[name]["ground_truths"]))
    return out


def evaluate_stratified(
    predictions: list[dict],
    ground_truths: list[np.ndarray],
    iou_thr: float = 0.5,
    strata=DEFAULT_STRATA,
) -> dict[str, dict]:
    """Compute MR^-2 / AP / JI per density stratum (plus overall)."""
    results = {}
    overall = evaluate(predictions, ground_truths, iou_thr=iou_thr)
    overall.update({"num_images": len(ground_truths),
                    "num_gt": int(sum(len(g) for g in ground_truths))})
    results["overall"] = overall

    buckets = stratify_by_density(predictions, ground_truths, strata)
    for name, bucket in buckets.items():
        if bucket["num_images"] == 0:
            results[name] = {"mr": None, "ap": None, "ji": None,
                             "num_images": 0, "num_gt": 0}
            continue
        m = evaluate(bucket["predictions"], bucket["ground_truths"], iou_thr=iou_thr)
        m.update({"num_images": bucket["num_images"], "num_gt": bucket["num_gt"]})
        results[name] = m
    return results


def format_stratified_table(results_by_model: dict[str, dict]) -> str:
    """Render a comparison table across models and strata as plain text."""
    strata_order = ["overall", "sparse", "medium", "dense"]
    lines = []
    header = f"{'stratum':<10} {'model':<20} {'imgs':>6} {'GT':>7} {'MR^-2':>8} {'AP':>8} {'JI':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for stratum in strata_order:
        for model_name, res in results_by_model.items():
            r = res.get(stratum)
            if r is None or r["num_images"] == 0:
                continue
            lines.append(
                f"{stratum:<10} {model_name:<20} {r['num_images']:>6} {r['num_gt']:>7} "
                f"{r['mr']:>8.4f} {r['ap']:>8.4f} {r['ji']:>8.4f}"
            )
        lines.append("")
    return "\n".join(lines)
