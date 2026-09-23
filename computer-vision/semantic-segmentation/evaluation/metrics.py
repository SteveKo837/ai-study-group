from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SegmentationScores:
    pixel_accuracy: float
    mean_iou: float
    dice_score: float

    def as_dict(self) -> dict[str, float]:
        return {
            "pixel_accuracy": self.pixel_accuracy,
            "mean_iou": self.mean_iou,
            "dice_score": self.dice_score,
        }


class SegmentationMetricAccumulator:
    """Accumulates a confusion matrix and computes common segmentation metrics."""

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        predictions_np = predictions.detach().cpu().numpy().astype(np.int64)
        targets_np = targets.detach().cpu().numpy().astype(np.int64)
        valid = targets_np != self.ignore_index
        valid &= (targets_np >= 0) & (targets_np < self.num_classes)
        labels = self.num_classes * targets_np[valid] + predictions_np[valid]
        counts = np.bincount(labels, minlength=self.num_classes**2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> SegmentationScores:
        true_positive = np.diag(self.confusion).astype(np.float64)
        support = self.confusion.sum(axis=1).astype(np.float64)
        predicted = self.confusion.sum(axis=0).astype(np.float64)

        pixel_accuracy = true_positive.sum() / max(self.confusion.sum(), 1)
        union = support + predicted - true_positive
        valid_classes = union > 0
        iou = np.divide(true_positive, union, out=np.zeros_like(true_positive), where=valid_classes)
        dice_denominator = support + predicted
        dice = np.divide(
            2 * true_positive,
            dice_denominator,
            out=np.zeros_like(true_positive),
            where=dice_denominator > 0,
        )
        return SegmentationScores(
            pixel_accuracy=float(pixel_accuracy),
            mean_iou=float(iou[valid_classes].mean() if valid_classes.any() else 0.0),
            dice_score=float(dice[valid_classes].mean() if valid_classes.any() else 0.0),
        )
