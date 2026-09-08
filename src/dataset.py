from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a mapping.")

    return config


def build_transforms(
    config: dict[str, Any],
) -> tuple[transforms.Compose, transforms.Compose]:
    height, width, channels = map(
        int,
        config["model"]["input_shape"],
    )

    if channels != 3:
        raise ValueError(
            "The dataset expects RGB images with 3 channels."
        )

    augmentation = config["data"].get(
        "augmentation",
        {},
    )

    train_transforms: list[Any] = [
        transforms.Resize((height, width)),
    ]

    if augmentation.get("enabled", True):
        if augmentation.get("horizontal_flip", True):
            train_transforms.append(
                transforms.RandomHorizontalFlip()
            )

        rotation = float(
            augmentation.get("rotation", 0.0)
        )

        if rotation > 0:
            train_transforms.append(
                transforms.RandomRotation(
                    degrees=rotation
                )
            )

        zoom = float(
            augmentation.get("zoom", 0.0)
        )

        if zoom > 0:
            train_transforms.append(
                transforms.RandomAffine(
                    degrees=0,
                    scale=(
                        max(1.0 - zoom, 0.0),
                        1.0 + zoom,
                    ),
                )
            )

        contrast = float(
            augmentation.get("contrast", 0.0)
        )

        if contrast > 0:
            train_transforms.append(
                transforms.ColorJitter(
                    contrast=contrast
                )
            )

    train_transforms.append(
        transforms.ToTensor()
    )

    eval_transforms = [
        transforms.Resize((height, width)),
        transforms.ToTensor(),
    ]

    return (
        transforms.Compose(train_transforms),
        transforms.Compose(eval_transforms),
    )


def build_datasets(
    config: dict[str, Any],
) -> tuple[
    Subset,
    Subset,
    Subset,
    list[str],
]:
    raw_dir = Path(config["data"]["raw_dir"])

    if not raw_dir.is_absolute():
        raw_dir = PROJECT_ROOT / raw_dir

    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {raw_dir}"
        )

    train_transform, eval_transform = build_transforms(
        config
    )

    # This dataset is used only to discover classes and
    # generate deterministic split indices.
    base_dataset = datasets.ImageFolder(
        raw_dir,
        transform=eval_transform,
    )

    total_size = len(base_dataset)

    if total_size == 0:
        raise ValueError(
            f"No images found in dataset: {raw_dir}"
        )

    train_split = float(
        config["data"]["train_split"]
    )

    validation_split = float(
        config["data"]["validation_split"]
    )

    test_split = float(
        config["data"]["test_split"]
    )

    split_total = (
        train_split
        + validation_split
        + test_split
    )

    if abs(split_total - 1.0) > 1e-6:
        raise ValueError(
            "train_split + validation_split + "
            "test_split must equal 1.0."
        )

    train_size = int(
        total_size * train_split
    )

    validation_size = int(
        total_size * validation_split
    )

    test_size = (
        total_size
        - train_size
        - validation_size
    )

    if min(
        train_size,
        validation_size,
        test_size,
    ) <= 0:
        raise ValueError(
            "Each dataset split must contain at least one image."
        )

    generator = torch.Generator().manual_seed(
        int(config["data"]["random_seed"])
    )

    train_base, validation_base, test_base = (
        torch.utils.data.random_split(
            base_dataset,
            [
                train_size,
                validation_size,
                test_size,
            ],
            generator=generator,
        )
    )

    # Use a separate ImageFolder for training so that
    # augmentation is applied only to training samples.
    train_source = datasets.ImageFolder(
        raw_dir,
        transform=train_transform,
    )

    train_dataset = Subset(
        train_source,
        train_base.indices,
    )

    validation_dataset = Subset(
        base_dataset,
        validation_base.indices,
    )

    test_dataset = Subset(
        base_dataset,
        test_base.indices,
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        base_dataset.classes,
    )


def build_dataloaders(
    config: dict[str, Any],
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    list[str],
]:
    (
        train_dataset,
        validation_dataset,
        test_dataset,
        classes,
    ) = build_datasets(config)

    batch_size = int(
        config["data"]["batch_size"]
    )

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": 2,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **common_kwargs,
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **common_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_kwargs,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        classes,
    )


def get_dataset_info(
    config: dict[str, Any],
) -> dict[str, Any]:
    raw_dir = Path(config["data"]["raw_dir"])

    if not raw_dir.is_absolute():
        raw_dir = PROJECT_ROOT / raw_dir

    dataset = datasets.ImageFolder(raw_dir)

    class_counts = {
        class_name: 0
        for class_name in dataset.classes
    }

    for _, class_index in dataset.samples:
        class_name = dataset.classes[class_index]
        class_counts[class_name] += 1

    return {
        "classes": dataset.classes,
        "num_classes": len(dataset.classes),
        "total_images": len(dataset),
        "class_counts": class_counts,
    }
