from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")

from transformers import SegformerForSemanticSegmentation

from datasets.voc_dataset import build_dataloaders
from evaluation.metrics import SegmentationMetricAccumulator
from utils.config import ensure_dir, load_config
from utils.seed import set_seed
from utils.voc import VOC_CLASSES
from visualizations.masks import save_prediction_triplet
from visualizations.plots import save_training_curves


def history_row(epoch: int, train_loss: float, val_loss: float, train_scores: dict, val_scores: dict) -> dict:
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_pixel_accuracy": train_scores["pixel_accuracy"],
        "val_pixel_accuracy": val_scores["pixel_accuracy"],
        "train_mean_iou": train_scores["mean_iou"],
        "val_mean_iou": val_scores["mean_iou"],
        "train_dice_score": train_scores["dice_score"],
        "val_dice_score": val_scores["dice_score"],
    }


def class_names_for_config(config: dict) -> list[str]:
    if config["data"].get("dataset", "voc2012").lower() in {"oxford_pet", "oxford-iiit-pet", "pet"}:
        return ["background", "pet"]
    return VOC_CLASSES[: config["project"]["num_classes"]]


def build_model(config: dict) -> SegformerForSemanticSegmentation:
    id2label = {index: name for index, name in enumerate(class_names_for_config(config))}
    label2id = {name: index for index, name in id2label.items()}
    model = SegformerForSemanticSegmentation.from_pretrained(
        config["transformer"]["model_name"],
        num_labels=config["project"]["num_classes"],
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.config.semantic_loss_ignore_index = config["project"]["ignore_index"]
    return model


def resize_logits(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    return nn.functional.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)


def run_epoch(model, dataloader, optimizer, device, config) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    metric = SegmentationMetricAccumulator(config["project"]["num_classes"], config["project"]["ignore_index"])
    total_loss = 0.0
    total_items = 0

    for images, masks in tqdm(dataloader, leave=False):
        images, masks = images.to(device), masks.to(device)
        with torch.set_grad_enabled(training):
            output = model(pixel_values=images, labels=masks)
            loss = output.loss
            logits = resize_logits(output.logits, masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        metric.update(logits.argmax(dim=1), masks)
        total_loss += loss.item() * images.size(0)
        total_items += images.size(0)
    return total_loss / max(total_items, 1), metric.compute().as_dict()


@torch.no_grad()
def save_examples(model, dataloader, device, output_dir: Path, limit: int = 8) -> None:
    model.eval()
    saved = 0
    for images, masks in dataloader:
        output = model(pixel_values=images.to(device))
        logits = resize_logits(output.logits, masks.to(device))
        predictions = logits.argmax(dim=1).cpu()
        for image, mask, prediction in zip(images, masks, predictions):
            save_prediction_triplet(
                image,
                mask,
                prediction,
                output_dir / f"segformer_sample_{saved:03d}.png",
                "SegFormer prediction",
            )
            saved += 1
            if saved >= limit:
                return


def main() -> None:
    config = load_config()
    set_seed(config["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = build_dataloaders(config)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["transformer"]["learning_rate"],
        weight_decay=config["transformer"]["weight_decay"],
    )

    early_cfg = config["transformer"].get("early_stopping", {})
    patience = int(early_cfg.get("patience", 5))
    min_delta = float(early_cfg.get("min_delta", 0.0))
    best_val_dice = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    checkpoint_dir = ensure_dir(config["transformer"]["checkpoint_dir"])
    output_root = Path(config["project"]["output_dir"])
    history_path = ensure_dir(output_root / "metrics") / "segformer_history.csv"
    print(f"SegFormer early stopping: patience={patience}, monitor=val_dice_score, min_delta={min_delta}")

    for epoch in range(1, config["transformer"]["epochs"] + 1):
        train_loss, train_scores = run_epoch(model, dataloaders["train"], optimizer, device, config)
        val_loss, val_scores = run_epoch(model, dataloaders["val"], None, device, config)
        history.append(history_row(epoch, train_loss, val_loss, train_scores, val_scores))
        pd.DataFrame(history).to_csv(history_path, index=False)

        improved = val_scores["dice_score"] > best_val_dice + min_delta
        if improved:
            best_val_dice = val_scores["dice_score"]
            best_epoch = epoch
            epochs_without_improvement = 0
            model.save_pretrained(checkpoint_dir)
        else:
            epochs_without_improvement += 1

        print(
            f"epoch {epoch}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_mIoU={train_scores['mean_iou']:.4f}, val_mIoU={val_scores['mean_iou']:.4f}, "
            f"train_dice={train_scores['dice_score']:.4f}, val_dice={val_scores['dice_score']:.4f}, "
            f"best_val_dice={best_val_dice:.4f} at epoch {best_epoch}, "
            f"no_improve={epochs_without_improvement}/{patience}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping SegFormer at epoch {epoch}; best validation Dice was {best_val_dice:.4f} at epoch {best_epoch}.")
            break

    pd.DataFrame(history).to_csv(history_path, index=False)
    save_training_curves(history, output_root / "figures" / "segformer_training_curves.png")

    model = SegformerForSemanticSegmentation.from_pretrained(checkpoint_dir).to(device)
    rows = []
    for split, dataloader in dataloaders.items():
        loss, scores = run_epoch(model, dataloader, None, device, config)
        rows.append({"method": "segformer", "split": split, "loss": loss, **scores})
    pd.DataFrame(rows).to_csv(ensure_dir(output_root / "metrics") / "segformer_metrics.csv", index=False)
    save_examples(model, dataloaders["test"], device, output_root / "segformer" / "examples")


if __name__ == "__main__":
    main()
