from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from datasets.voc_dataset import build_dataloaders
from unet.model import UNet
from utils.config import load_config
from visualizations.masks import save_prediction_triplet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run U-Net inference visualizations on a dataset split.")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to visualize.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help="Number of examples to save. Use -1 to save the full split.",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(config["unet"]["checkpoint"], map_location=device)
    model = UNet(config["project"]["num_classes"], config["unet"]["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataloader = build_dataloaders(config)[args.split]
    output_dir = Path(config["project"]["output_dir"]) / "unet" / "inference" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    limit = None if args.limit < 0 else args.limit
    total = len(dataloader.dataset) if limit is None else min(len(dataloader.dataset), limit)
    print(f"Running U-Net inference on {args.split}; saving {total} examples to {output_dir}")

    saved = 0
    with tqdm(total=total, desc=f"unet inference {args.split}", unit="image") as progress:
        for images, masks in dataloader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            for image, mask, prediction in zip(images, masks, predictions):
                save_prediction_triplet(
                    image,
                    mask,
                    prediction,
                    output_dir / f"sample_{saved:03d}.png",
                    f"U-Net inference: {args.split}",
                )
                saved += 1
                progress.update(1)
                if limit is not None and saved >= limit:
                    return


if __name__ == "__main__":
    main()
