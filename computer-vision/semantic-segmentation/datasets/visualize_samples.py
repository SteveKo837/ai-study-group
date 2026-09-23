from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from datasets.voc_dataset import build_dataloaders
from utils.config import ensure_dir, load_config
from visualizations.masks import colorize_mask, image_tensor_to_numpy


def main() -> None:
    config = load_config()
    output_dir = ensure_dir(Path(config["project"]["output_dir"]) / "dataset_samples")
    dataloader = build_dataloaders(config)["train"]
    images, masks = next(iter(dataloader))

    count = min(6, images.size(0))
    fig, axes = plt.subplots(count, 2, figsize=(7, 3 * count))
    if count == 1:
        axes = axes[None, :]
    for index in range(count):
        axes[index, 0].imshow(image_tensor_to_numpy(images[index]))
        axes[index, 0].set_title("image")
        axes[index, 0].axis("off")
        axes[index, 1].imshow(colorize_mask(masks[index]))
        axes[index, 1].set_title("semantic mask")
        axes[index, 1].axis("off")
    fig.tight_layout()
    path = output_dir / "voc_samples.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
