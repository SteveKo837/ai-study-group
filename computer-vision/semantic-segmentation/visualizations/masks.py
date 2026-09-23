from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.voc_dataset import denormalize_image
from utils.config import ensure_dir
from utils.voc import VOC_PALETTE


def colorize_mask(mask: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert class-ID mask to an RGB visualization using the VOC palette."""
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    palette = np.array(VOC_PALETTE, dtype=np.uint8)
    safe_mask = np.where((mask >= 0) & (mask < len(palette)), mask, 0)
    rgb = palette[safe_mask]
    rgb[mask == 255] = np.array([224, 224, 224], dtype=np.uint8)
    return rgb


def image_tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    image = denormalize_image(image.detach().cpu())
    return image.permute(1, 2, 0).numpy()


def save_prediction_triplet(
    image: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    output_path: str | Path,
    title: str,
) -> None:
    ensure_dir(Path(output_path).parent)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        (image_tensor_to_numpy(image), "input image"),
        (colorize_mask(target), "ground truth"),
        (colorize_mask(prediction), "prediction"),
    ]
    for axis, (panel, panel_title) in zip(axes, panels):
        axis.imshow(panel)
        axis.set_title(panel_title)
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_comparison_grid(
    rows: list[dict[str, np.ndarray]],
    column_names: list[str],
    output_path: str | Path,
) -> None:
    ensure_dir(Path(output_path).parent)
    fig, axes = plt.subplots(len(rows), len(column_names), figsize=(3.2 * len(column_names), 3.2 * len(rows)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(column_names):
            axes[row_index, column_index].imshow(row[column])
            axes[row_index, column_index].set_title(column)
            axes[row_index, column_index].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
