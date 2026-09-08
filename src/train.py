import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.dataset import build_dataloaders
from src.model import build_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(
    path: str = "config.yaml",
) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration file must contain a mapping."
        )

    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)
        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    if total == 0:
        raise ValueError(
            "Training dataset is empty."
        )

    return (
        total_loss / total,
        correct / total,
    )


@torch.no_grad()
def evaluate_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        batch_size = images.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    if total == 0:
        raise ValueError(
            "Validation dataset is empty."
        )

    return (
        total_loss / total,
        correct / total,
    )


def save_metrics(
    path: Path,
    history: list[dict[str, float]],
) -> None:
    if not history:
        return

    fieldnames = list(history[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    config = load_config()

    seed = int(
        config["data"]["random_seed"]
    )

    set_seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    (
        train_loader,
        validation_loader,
        _,
        classes,
    ) = build_dataloaders(config)

    print(f"Classes: {classes}")

    model = build_model(config)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    learning_rate = float(
        config["training"]["learning_rate"]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    training_config = config["training"]

    reduce_lr_config = training_config.get(
        "reduce_lr",
        {},
    )

    scheduler = None

    if reduce_lr_config.get(
        "enabled",
        False,
    ):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(
                reduce_lr_config.get(
                    "factor",
                    0.5,
                )
            ),
            patience=int(
                reduce_lr_config.get(
                    "patience",
                    4,
                )
            ),
            min_lr=float(
                reduce_lr_config.get(
                    "min_lr",
                    1e-5,
                )
            ),
        )

    output_dir = Path(
        config["environment"]["output_dir"]
    )

    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = int(
        training_config["epochs"]
    )

    early_stopping_config = (
        training_config.get(
            "early_stopping",
            {},
        )
    )

    early_stopping_enabled = (
        early_stopping_config.get(
            "enabled",
            False,
        )
    )

    early_stopping_patience = int(
        early_stopping_config.get(
            "patience",
            10,
        )
    )

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    history: list[dict[str, float]] = []

    best_model_path = (
        output_dir /
        "best_model.pt"
    )

    final_model_path = (
        output_dir /
        "final_model.pt"
    )

    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        validation_loss, validation_accuracy = (
            evaluate_epoch(
                model,
                validation_loader,
                criterion,
                device,
            )
        )

        if scheduler is not None:
            scheduler.step(
                validation_loss
            )

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        epoch_metrics = {
            "epoch": float(epoch + 1),
            "loss": train_loss,
            "accuracy": train_accuracy,
            "val_loss": validation_loss,
            "val_accuracy": validation_accuracy,
            "learning_rate": current_learning_rate,
        }

        history.append(epoch_metrics)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- loss={train_loss:.4f} "
            f"acc={train_accuracy:.4f} "
            f"val_loss={validation_loss:.4f} "
            f"val_acc={validation_accuracy:.4f} "
            f"lr={current_learning_rate:.6g}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                best_model_path,
            )

            print(
                f"  Saved best model: {best_model_path}"
            )

        else:
            epochs_without_improvement += 1

        if (
            early_stopping_enabled
            and epochs_without_improvement
            >= early_stopping_patience
        ):
            print(
                f"Early stopping after "
                f"{epoch + 1} epochs."
            )
            break

    torch.save(
        model.state_dict(),
        final_model_path,
    )

    metrics_path = (
        output_dir /
        "training_metrics.csv"
    )

    save_metrics(
        metrics_path,
        history,
    )

    print(
        f"Final model saved to: {final_model_path}"
    )

    print(
        f"Training metrics saved to: {metrics_path}"
    )

    print("Training complete.")


if __name__ == "__main__":
    main()
