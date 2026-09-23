from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils.config import ensure_dir


def save_training_curves(history: list[dict], output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train loss")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(frame["epoch"], frame["val_mean_iou"], label="val mIoU", color="#1f77b4")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_iou_bar_chart(results_csv: str | Path, output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    frame = pd.read_csv(results_csv)
    pivot = frame.pivot(index="method", columns="split", values="mean_iou")
    axis = pivot.plot(kind="bar", figsize=(9, 5), rot=0)
    axis.set_ylabel("mean IoU")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.3)
    axis.figure.tight_layout()
    axis.figure.savefig(output_path, dpi=170)
    plt.close(axis.figure)
