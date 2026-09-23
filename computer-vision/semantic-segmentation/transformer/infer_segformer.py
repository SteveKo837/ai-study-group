from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch
from tqdm import tqdm

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")

from transformers import SegformerForSemanticSegmentation

from datasets.voc_dataset import build_dataloaders
from transformer.train_segformer import resize_logits
from utils.config import load_config
from visualizations.masks import save_prediction_triplet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SegFormer inference visualizations on a dataset split.")
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
    model = SegformerForSemanticSegmentation.from_pretrained(config["transformer"]["checkpoint_dir"]).to(device)
    model.eval()

    dataloader = build_dataloaders(config)[args.split]
    output_dir = Path(config["project"]["output_dir"]) / "segformer" / "inference" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    limit = None if args.limit < 0 else args.limit
    total = len(dataloader.dataset) if limit is None else min(len(dataloader.dataset), limit)
    print(f"Running SegFormer inference on {args.split}; saving {total} examples to {output_dir}")

    saved = 0
    with tqdm(total=total, desc=f"segformer inference {args.split}", unit="image") as progress:
        for images, masks in dataloader:
            output = model(pixel_values=images.to(device))
            logits = resize_logits(output.logits, masks.to(device))
            predictions = logits.argmax(dim=1).cpu()
            for image, mask, prediction in zip(images, masks, predictions):
                save_prediction_triplet(
                    image,
                    mask,
                    prediction,
                    output_dir / f"sample_{saved:03d}.png",
                    f"SegFormer inference: {args.split}",
                )
                saved += 1
                progress.update(1)
                if limit is not None and saved >= limit:
                    return


if __name__ == "__main__":
    main()
