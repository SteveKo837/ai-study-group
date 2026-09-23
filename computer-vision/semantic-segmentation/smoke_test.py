from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from evaluation.metrics import SegmentationMetricAccumulator
from traditional_cv.classical_pipeline import watershed_segment
from unet.model import UNet
from utils.config import ensure_dir, load_config
from utils.seed import set_seed
from visualizations.masks import save_prediction_triplet


def make_synthetic_batch(num_samples: int = 4, image_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a small VOC-shaped batch for local execution checks."""
    images = torch.rand(num_samples, 3, image_size, image_size)
    masks = torch.zeros(num_samples, image_size, image_size, dtype=torch.long)
    masks[:, 12:42, 10:35] = 15
    masks[:, 28:55, 34:58] = 7
    return images, masks


def smoke_traditional(config: dict, images: torch.Tensor, masks: torch.Tensor) -> None:
    metric = SegmentationMetricAccumulator(config["project"]["num_classes"], config["project"]["ignore_index"])
    output_dir = ensure_dir(Path(config["project"]["output_dir"]) / "smoke_test")
    image_rgb = (images[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    prediction, _ = watershed_segment(image_rgb, config)
    prediction_tensor = torch.as_tensor(prediction, dtype=torch.long)
    metric.update(prediction_tensor.unsqueeze(0), masks[0].unsqueeze(0))
    save_prediction_triplet(images[0], masks[0], prediction_tensor, output_dir / "traditional_smoke.png", "Traditional CV smoke")
    print("traditional CV smoke:", metric.compute().as_dict())


def smoke_unet(config: dict, images: torch.Tensor, masks: torch.Tensor) -> None:
    model = UNet(config["project"]["num_classes"], base_channels=8)
    criterion = nn.CrossEntropyLoss(ignore_index=config["project"]["ignore_index"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    logits = model(images)
    loss = criterion(logits, masks)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    prediction = logits.argmax(dim=1)[0]
    save_prediction_triplet(
        images[0],
        masks[0],
        prediction,
        Path(config["project"]["output_dir"]) / "smoke_test" / "unet_smoke.png",
        "U-Net smoke",
    )
    print("U-Net smoke loss:", round(float(loss.item()), 4))


def smoke_segformer(config: dict, images: torch.Tensor, masks: torch.Tensor) -> None:
    tiny_config = SegformerConfig(
        num_labels=config["project"]["num_classes"],
        num_encoder_blocks=4,
        depths=[1, 1, 1, 1],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[8, 16, 32, 64],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 1, 2, 4],
        decoder_hidden_size=32,
        semantic_loss_ignore_index=config["project"]["ignore_index"],
    )
    model = SegformerForSemanticSegmentation(tiny_config)
    output = model(pixel_values=images, labels=masks)
    print("tiny SegFormer smoke loss:", round(float(output.loss.item()), 4))


def main() -> None:
    config = load_config()
    set_seed(config["project"]["seed"])
    images, masks = make_synthetic_batch()
    DataLoader(TensorDataset(images, masks), batch_size=2)
    smoke_traditional(config, images, masks)
    smoke_unet(config, images, masks)
    smoke_segformer(config, images, masks)
    print(f"smoke test images saved to {Path(config['project']['output_dir']) / 'smoke_test'}")


if __name__ == "__main__":
    main()
