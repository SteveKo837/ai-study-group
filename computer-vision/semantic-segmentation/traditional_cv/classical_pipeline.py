from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from tqdm import tqdm

from datasets.voc_dataset import build_dataloaders, denormalize_image
from evaluation.metrics import SegmentationMetricAccumulator
from utils.config import ensure_dir, load_config
from utils.seed import set_seed
from visualizations.masks import colorize_mask

matplotlib.use("Agg")


def watershed_segment(image_rgb: np.ndarray, config: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """A simple hand-crafted segmentation pipeline.

    This intentionally does not know VOC classes. It finds low-level image
    regions, then maps every foreground-looking region to the generic VOC
    "person" class so metric failure is easy to discuss in a live demo.
    """
    cv_cfg = config["traditional_cv"]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (cv_cfg["blur_kernel"], cv_cfg["blur_kernel"]), 0)

    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobel_x, sobel_y)
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edge_binary = cv2.threshold(edges, cv_cfg["sobel_threshold"], 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    closed_edges = cv2.morphologyEx(edge_binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    sure_background = cv2.dilate(closed_edges, kernel, iterations=3)
    distance = cv2.distanceTransform(255 - closed_edges, cv2.DIST_L2, 5)
    _, sure_foreground = cv2.threshold(distance, 0.35 * distance.max(), 255, 0)
    sure_foreground = sure_foreground.astype(np.uint8)
    unknown = cv2.subtract(sure_background, sure_foreground)

    _, markers = cv2.connectedComponents(sure_foreground)
    markers = markers + 1
    markers[unknown == 255] = 0
    watershed_markers = cv2.watershed(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), markers.copy())

    prediction = np.zeros(gray.shape, dtype=np.int64)
    foreground_class = 15 if config["project"]["num_classes"] > 2 else 1
    for marker_id in np.unique(watershed_markers):
        if marker_id <= 1:
            continue
        component = watershed_markers == marker_id
        if int(component.sum()) >= cv_cfg["min_component_area"]:
            prediction[component] = foreground_class

    intermediate = {
        "original": image_rgb,
        "edge_map": edges,
        "watershed": colorize_regions(watershed_markers),
        "final": colorize_mask(prediction),
    }
    return prediction, intermediate


def colorize_regions(markers: np.ndarray) -> np.ndarray:
    normalized = markers.astype(np.float32)
    normalized[normalized < 0] = 0
    if normalized.max() > 0:
        normalized = normalized / normalized.max()
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def save_intermediate_panel(intermediate: dict[str, np.ndarray], target: torch.Tensor, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    panels = [
        ("original", intermediate["original"], None),
        ("edge map", intermediate["edge_map"], "gray"),
        ("watershed regions", intermediate["watershed"], None),
        ("ground truth", colorize_mask(target), None),
        ("final prediction", intermediate["final"], None),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(17, 4))
    for axis, (title, image, cmap) in zip(axes, panels):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def evaluate_split(split: str, dataloader, config: dict) -> dict[str, float]:
    metric = SegmentationMetricAccumulator(config["project"]["num_classes"], config["project"]["ignore_index"])
    output_dir = ensure_dir(Path(config["project"]["output_dir"]) / "traditional_cv" / split)
    max_images = config["traditional_cv"].get("max_images_per_split")
    total_images = len(dataloader.dataset)
    progress_total = min(total_images, max_images) if max_images is not None else total_images
    seen = 0

    mode = f"first {progress_total} images" if max_images is not None else "full split"
    print(f"[traditional_cv] {split}: evaluating {mode}; saving examples to {output_dir}")

    with tqdm(total=progress_total, desc=f"traditional cv {split}", unit="image") as progress:
        for images, masks in dataloader:
            for image_tensor, mask in zip(images, masks):
                image_rgb = (denormalize_image(image_tensor).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                prediction, intermediate = watershed_segment(image_rgb, config)
                prediction_tensor = torch.as_tensor(prediction, dtype=torch.long)
                metric.update(prediction_tensor.unsqueeze(0), mask.unsqueeze(0))
                if seen < 8:
                    save_intermediate_panel(intermediate, mask, output_dir / f"sample_{seen:03d}_pipeline.png")
                seen += 1
                progress.update(1)
                progress.set_postfix_str(f"saved_examples={min(seen, 8)}/8")
                if max_images is not None and seen >= max_images:
                    return metric.compute().as_dict()
    return metric.compute().as_dict()


def main() -> None:
    config = load_config()
    set_seed(config["project"]["seed"])
    dataloaders = build_dataloaders(config)
    rows = []
    for split, dataloader in dataloaders.items():
        scores = evaluate_split(split, dataloader, config)
        rows.append({"method": "traditional_cv", "split": split, **scores})
        print(
            f"[traditional_cv] {split}: "
            f"pixel_acc={scores['pixel_accuracy']:.4f}, "
            f"mIoU={scores['mean_iou']:.4f}, "
            f"dice={scores['dice_score']:.4f}"
        )
    import pandas as pd

    output_path = ensure_dir(Path(config["project"]["output_dir"]) / "metrics") / "traditional_cv_metrics.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
