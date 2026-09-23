from __future__ import annotations

from datasets.voc_dataset import build_segmentation_datasets
from utils.config import load_config


def main() -> None:
    config = load_config()
    datasets = build_segmentation_datasets(config)
    dataset_name = config["data"].get("dataset", "voc2012")
    for split, dataset in datasets.items():
        print(f"{split}: {len(dataset)} labeled examples")
    print(f"{dataset_name} data is ready.")


if __name__ == "__main__":
    main()
