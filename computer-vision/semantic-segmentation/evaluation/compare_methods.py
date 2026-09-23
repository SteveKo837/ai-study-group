from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import SegformerForSemanticSegmentation

from datasets.voc_dataset import build_dataloaders, denormalize_image
from traditional_cv.classical_pipeline import watershed_segment
from transformer.train_segformer import resize_logits
from unet.model import UNet
from utils.config import ensure_dir, load_config
from visualizations.masks import colorize_mask, save_comparison_grid
from visualizations.plots import save_iou_bar_chart


def collect_metric_files(output_root: Path) -> pd.DataFrame:
    metric_dir = output_root / "metrics"
    frames = []
    for filename in ["traditional_cv_metrics.csv", "unet_metrics.csv", "segformer_metrics.csv"]:
        path = metric_dir / filename
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No metrics found. Run the training/evaluation scripts first.")
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(metric_dir / "all_methods_metrics.csv", index=False)
    (metric_dir / "all_methods_metrics.md").write_text(frame.to_markdown(index=False), encoding="utf-8")
    return frame


def load_unet_if_available(config: dict, device: torch.device):
    checkpoint = Path(config["unet"]["checkpoint"])
    if not checkpoint.exists():
        return None
    model = UNet(config["project"]["num_classes"], config["unet"]["base_channels"]).to(device)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def load_segformer_if_available(config: dict, device: torch.device):
    checkpoint_dir = Path(config["transformer"]["checkpoint_dir"])
    if not checkpoint_dir.exists():
        return None
    model = SegformerForSemanticSegmentation.from_pretrained(checkpoint_dir).to(device)
    model.eval()
    return model


@torch.no_grad()
def save_qualitative_comparison(config: dict, output_root: Path, limit: int = 6) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = build_dataloaders(config)["test"]
    unet = load_unet_if_available(config, device)
    segformer = load_segformer_if_available(config, device)
    rows = []

    for images, masks in dataloader:
        for image, mask in zip(images, masks):
            image_rgb = (denormalize_image(image).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            traditional_prediction, _ = watershed_segment(image_rgb, config)
            row = {
                "image": image_rgb,
                "ground truth": colorize_mask(mask),
                "traditional CV": colorize_mask(traditional_prediction),
                "U-Net": np.full_like(image_rgb, 235),
                "SegFormer": np.full_like(image_rgb, 235),
            }
            if unet is not None:
                pred = unet(image.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu()
                row["U-Net"] = colorize_mask(pred)
            if segformer is not None:
                output = segformer(pixel_values=image.unsqueeze(0).to(device))
                logits = resize_logits(output.logits, mask.unsqueeze(0).to(device))
                pred = logits.argmax(dim=1).squeeze(0).cpu()
                row["SegFormer"] = colorize_mask(pred)
            rows.append(row)
            if len(rows) >= limit:
                columns = ["image", "ground truth", "traditional CV", "U-Net", "SegFormer"]
                save_comparison_grid(rows, columns, output_root / "figures" / "qualitative_comparison_grid.png")
                return


def main() -> None:
    config = load_config()
    output_root = Path(config["project"]["output_dir"])
    metric_dir = ensure_dir(output_root / "metrics")
    frame = collect_metric_files(output_root)
    save_iou_bar_chart(metric_dir / "all_methods_metrics.csv", output_root / "figures" / "iou_comparison.png")
    save_qualitative_comparison(config, output_root)
    print(frame.to_markdown(index=False))
    print(f"saved comparison outputs under {output_root}")


if __name__ == "__main__":
    main()
