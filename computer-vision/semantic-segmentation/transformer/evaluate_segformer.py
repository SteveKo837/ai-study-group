from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import torch

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")

from transformers import SegformerForSemanticSegmentation

from datasets.voc_dataset import build_dataloaders
from evaluation.metrics import SegmentationMetricAccumulator
from transformer.train_segformer import resize_logits
from utils.config import ensure_dir, load_config


@torch.no_grad()
def evaluate_split(model, dataloader, device, config) -> dict[str, float]:
    model.eval()
    metric = SegmentationMetricAccumulator(config["project"]["num_classes"], config["project"]["ignore_index"])
    total_loss = 0.0
    total_items = 0
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        output = model(pixel_values=images, labels=masks)
        logits = resize_logits(output.logits, masks)
        metric.update(logits.argmax(dim=1), masks)
        total_loss += output.loss.item() * images.size(0)
        total_items += images.size(0)
    return {"loss": total_loss / max(total_items, 1), **metric.compute().as_dict()}


def main() -> None:
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained(config["transformer"]["checkpoint_dir"]).to(device)
    rows = []
    for split, dataloader in build_dataloaders(config).items():
        rows.append({"method": "segformer", "split": split, **evaluate_split(model, dataloader, device, config)})
    output = ensure_dir(Path(config["project"]["output_dir"]) / "metrics") / "segformer_metrics.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
