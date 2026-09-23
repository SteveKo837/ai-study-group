from __future__ import annotations

from datasets.voc_dataset import build_segmentation_datasets
from utils.config import load_config


def main() -> None:
    config = load_config()
    datasets = build_segmentation_datasets(config)
    for split, dataset in datasets.items():
        print(f"{split}: {len(dataset)} labeled examples")
    print("Dataset is ready. Use datasets.prepare_dataset for the generic entry point.")


if __name__ == "__main__":
    main()
