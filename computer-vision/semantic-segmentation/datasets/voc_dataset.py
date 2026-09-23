from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import OxfordIIITPet, VOCSegmentation
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class VOCBatch:
    images: torch.Tensor
    masks: torch.Tensor


class JointVOCTransform:
    """Resize image and mask together while preserving mask class IDs."""

    def __init__(self, image_size: int, train: bool = False, augmentation: dict | None = None) -> None:
        self.image_size = image_size
        self.train = train
        self.augmentation = augmentation or {}

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if self.train:
            image, mask = self._augment(image, mask)

        image = TF.resize(image, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        mask_tensor = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image_tensor, mask_tensor

    def _augment(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        """Apply spatial augmentations identically to image and mask."""
        if self.augmentation.get("horizontal_flip", True) and torch.rand(1).item() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        max_rotation = float(self.augmentation.get("rotation_degrees", 0))
        if max_rotation > 0:
            angle = float(torch.empty(1).uniform_(-max_rotation, max_rotation).item())
            image = TF.rotate(
                image,
                angle,
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.rotate(
                mask,
                angle,
                interpolation=TF.InterpolationMode.NEAREST,
                fill=255,
            )

        scale_min = float(self.augmentation.get("scale_min", 1.0))
        scale_max = float(self.augmentation.get("scale_max", 1.0))
        if scale_min != 1.0 or scale_max != 1.0:
            image, mask = self._scale_crop_or_pad(image, mask, scale_min, scale_max)

        return image, mask

    def _scale_crop_or_pad(
        self,
        image: Image.Image,
        mask: Image.Image,
        scale_min: float,
        scale_max: float,
    ) -> tuple[Image.Image, Image.Image]:
        """Zoom in/out, then crop or pad to the model input size.

        scale > 1.0 zooms in because we resize larger and crop.
        scale < 1.0 zooms out because we resize smaller and pad.
        """
        scale = float(torch.empty(1).uniform_(scale_min, scale_max).item())
        scaled_size = max(1, int(round(self.image_size * scale)))
        image = TF.resize(image, [scaled_size, scaled_size], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [scaled_size, scaled_size], interpolation=TF.InterpolationMode.NEAREST)

        if scaled_size >= self.image_size:
            max_offset = scaled_size - self.image_size
            top = int(torch.randint(0, max_offset + 1, (1,)).item())
            left = int(torch.randint(0, max_offset + 1, (1,)).item())
            image = TF.crop(image, top, left, self.image_size, self.image_size)
            mask = TF.crop(mask, top, left, self.image_size, self.image_size)
        else:
            pad_total = self.image_size - scaled_size
            pad_left = int(torch.randint(0, pad_total + 1, (1,)).item())
            pad_top = int(torch.randint(0, pad_total + 1, (1,)).item())
            pad_right = pad_total - pad_left
            pad_bottom = pad_total - pad_top
            padding = [pad_left, pad_top, pad_right, pad_bottom]
            image = TF.pad(image, padding, fill=0)
            mask = TF.pad(mask, padding, fill=255)

        return image, mask


class JointPetTransform(JointVOCTransform):
    """Transform Oxford-IIIT Pet trimaps into semantic labels.

    Torchvision trimap values are:
    1 = pet foreground, 2 = background, 3 = border.
    We use a clean educational binary segmentation target:
    0 = background, 1 = pet, 255 = ignore border.
    """

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image_tensor, raw_mask = super().__call__(image, mask)
        mapped = torch.full_like(raw_mask, 255)
        mapped[raw_mask == 2] = 0
        mapped[raw_mask == 1] = 1
        return image_tensor, mapped


class VOCSegmentationWithTransform(Dataset):
    """Thin wrapper around torchvision VOC so image and mask transforms stay synchronized."""

    def __init__(
        self,
        root: str | Path,
        image_set: str,
        transform: Callable[[Image.Image, Image.Image], tuple[torch.Tensor, torch.Tensor]],
        download: bool,
    ) -> None:
        self.dataset = VOCSegmentation(
            root=str(root),
            year="2012",
            image_set=image_set,
            download=download,
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.dataset[index]
        return self.transform(image.convert("RGB"), mask)


class OxfordPetSegmentationWithTransform(Dataset):
    """Oxford-IIIT Pet segmentation dataset wrapper with synchronized transforms."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Callable[[Image.Image, Image.Image], tuple[torch.Tensor, torch.Tensor]],
        download: bool,
    ) -> None:
        self.dataset = OxfordIIITPet(
            root=str(root),
            split=split,
            target_types="segmentation",
            download=download,
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.dataset[index]
        return self.transform(image.convert("RGB"), mask)


def _split_indices(length: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(length, generator=generator).tolist()
    val_count = int(round(length * val_fraction))
    return indices[:val_count], indices[val_count:]


def build_voc_datasets(config: dict) -> dict[str, Dataset]:
    """Create train, validation, and local test datasets.

    VOC2012 semantic segmentation has public labels for train and val only.
    We keep the official train split intact, then split the labeled official
    val split into validation and local test partitions.
    """
    data_cfg = config["data"]
    seed = config["project"]["seed"]
    root = data_cfg["root"]
    image_size = data_cfg["image_size"]
    download = data_cfg.get("download", True)

    augmentation = data_cfg.get("augmentation", {})
    train_dataset = VOCSegmentationWithTransform(
        root,
        "train",
        JointVOCTransform(image_size, train=True, augmentation=augmentation),
        download,
    )
    official_val = VOCSegmentationWithTransform(root, "val", JointVOCTransform(image_size, train=False), download)
    val_indices, test_indices = _split_indices(
        len(official_val),
        data_cfg["val_fraction_from_official_val"],
        seed,
    )
    return {
        "train": train_dataset,
        "val": Subset(official_val, val_indices),
        "test": Subset(official_val, test_indices),
    }


def build_oxford_pet_datasets(config: dict) -> dict[str, Dataset]:
    """Create train, validation, and test datasets for Oxford-IIIT Pet.

    Oxford Pet has public trainval and test splits. We split trainval into
    train/val, then keep the official test split as test.
    """
    data_cfg = config["data"]
    seed = config["project"]["seed"]
    root = data_cfg["root"]
    image_size = data_cfg["image_size"]
    download = data_cfg.get("download", True)

    augmentation = data_cfg.get("augmentation", {})
    trainval = OxfordPetSegmentationWithTransform(
        root,
        "trainval",
        JointPetTransform(image_size, train=True, augmentation=augmentation),
        download,
    )
    test_dataset = OxfordPetSegmentationWithTransform(root, "test", JointPetTransform(image_size, train=False), download)
    val_indices, train_indices = _split_indices(
        len(trainval),
        data_cfg.get("val_fraction_from_trainval", 0.2),
        seed,
    )
    return {
        "train": Subset(trainval, train_indices),
        "val": Subset(trainval, val_indices),
        "test": test_dataset,
    }


def build_segmentation_datasets(config: dict) -> dict[str, Dataset]:
    dataset_name = config["data"].get("dataset", "voc2012").lower()
    if dataset_name in {"voc", "voc2012", "pascal_voc"}:
        return build_voc_datasets(config)
    if dataset_name in {"oxford_pet", "oxford-iiit-pet", "pet"}:
        return build_oxford_pet_datasets(config)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_dataloaders(config: dict) -> dict[str, DataLoader]:
    datasets = build_segmentation_datasets(config)
    data_cfg = config["data"]
    return {
        split: DataLoader(
            dataset,
            batch_size=data_cfg["batch_size"],
            shuffle=(split == "train"),
            num_workers=data_cfg["num_workers"],
            pin_memory=torch.cuda.is_available(),
        )
        for split, dataset in datasets.items()
    }


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device)[:, None, None]
    return (image * std + mean).clamp(0, 1)
