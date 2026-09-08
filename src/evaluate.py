"""
Evaluate the trained cattle classifier.

Evaluation is performed exclusively against the TEST split.

Generated artifacts include:

    - Accuracy
    - Precision
    - Recall
    - F1 score
    - Per-class metrics
    - Confusion matrix
    - Training history plots
    - JSON evaluation report

Usage:

    uv run python -m src.evaluate

Or specify a model:

    uv run python -m src.evaluate --model output/best_model.pt
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (
    build_dataloaders,
    get_dataset_info,
    load_config,
)
from src.model import build_model


MetricValue = int | float
EvaluationSummary = dict[str, float]
EvaluationReport = dict[str, Any]


def get_model_path(
    config: dict[str, Any],
    model_path: str | None,
) -> Path:
    """
    Resolve the model checkpoint path.

    Priority:

        1. Explicit --model argument
        2. Configured checkpoint path
        3. output/best_model.pt
        4. output/final_model.pt
    """

    if model_path:
        path = Path(model_path)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if not path.is_file():
            raise FileNotFoundError(
                f"Model not found: {path}"
            )

        return path

    training_config = config.get(
        "training",
        {},
    )

    checkpoint_config = training_config.get(
        "checkpointing",
        {},
    )

    checkpoint_path = checkpoint_config.get(
        "filepath"
    )

    if isinstance(
        checkpoint_path,
        str,
    ) and checkpoint_path:
        path = Path(checkpoint_path)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if path.is_file():
            return path

    output_dir = Path(
        config.get(
            "environment",
            {},
        ).get(
            "output_dir",
            "output",
        )
    )

    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    candidates = [
        output_dir / "best_model.pt",
        output_dir / "final_model.pt",
    ]

    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        "No trained model was found.\n"
        "Run train.py first or provide --model."
    )


@torch.inference_mode()
def predict_dataset(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions for the supplied dataset.

    Returns:

        y_true:
            Ground-truth class indices.

        y_pred:
            Predicted class indices.
    """

    model.eval()

    true_batches: list[np.ndarray] = []
    predicted_batches: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        true_batches.append(
            labels.cpu().numpy()
        )

        predicted_batches.append(
            predictions.cpu().numpy()
        )

    if not true_batches:
        raise ValueError(
            "The test dataset is empty."
        )

    y_true = np.concatenate(
        true_batches
    )

    y_pred = np.concatenate(
        predicted_batches
    )

    return y_true, y_pred


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> tuple[
    EvaluationSummary,
    pd.DataFrame,
]:
    """
    Calculate overall and per-class classification metrics.
    """

    labels = list(
        range(len(class_names))
    )

    accuracy = float(
        accuracy_score(
            y_true,
            y_pred,
        )
    )

    raw_report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # sklearn's type stubs do not precisely describe the dictionary
    # returned when output_dict=True. At runtime this is a dictionary
    # keyed by class names and aggregate metric names.
    report = cast(
        dict[str, dict[str, float | int]],
        raw_report,
    )

    macro_metrics = report["macro avg"]
    weighted_metrics = report["weighted avg"]

    summary: EvaluationSummary = {
        "accuracy": accuracy,
        "precision_macro": float(
            macro_metrics["precision"]
        ),
        "recall_macro": float(
            macro_metrics["recall"]
        ),
        "f1_macro": float(
            macro_metrics["f1-score"]
        ),
        "precision_weighted": float(
            weighted_metrics["precision"]
        ),
        "recall_weighted": float(
            weighted_metrics["recall"]
        ),
        "f1_weighted": float(
            weighted_metrics["f1-score"]
        ),
    }

    rows: list[
        dict[str, MetricValue | str]
    ] = []

    for class_name in class_names:
        metrics = report[class_name]

        rows.append(
            {
                "class": class_name,
                "precision": float(
                    metrics["precision"]
                ),
                "recall": float(
                    metrics["recall"]
                ),
                "f1_score": float(
                    metrics["f1-score"]
                ),
                "support": int(
                    metrics["support"]
                ),
            }
        )

    return (
        summary,
        pd.DataFrame(rows),
    )


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> np.ndarray:
    """
    Calculate and save the confusion matrix.
    """

    labels = list(
        range(len(class_names))
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    )

    path = (
        output_dir
        / "confusion_matrix.csv"
    )

    cm_df.to_csv(path)

    print(
        f"\nConfusion matrix saved to: {path}"
    )

    print("\nConfusion matrix:")
    print(cm_df)

    return cm


def analyze_training_history(
    output_dir: Path,
) -> dict[str, MetricValue] | None:
    """
    Analyze training_metrics.csv if it exists.
    """

    metrics_path = (
        output_dir
        / "training_metrics.csv"
    )

    if not metrics_path.is_file():
        print(
            "\nNo training_metrics.csv found."
        )
        return None

    metrics = pd.read_csv(
        metrics_path
    )

    if metrics.empty:
        print(
            "\nTraining metrics file is empty."
        )
        return None

    result: dict[str, MetricValue] = {
        "epochs": int(
            len(metrics)
        )
    }

    if "val_loss" in metrics.columns:
        val_loss = pd.to_numeric(
            metrics["val_loss"],
            errors="coerce",
        )

        if val_loss.notna().any():
            values = val_loss.to_numpy()

            position = int(
                np.nanargmin(values)
            )

            result["best_val_loss"] = float(
                val_loss.iloc[position]
            )

            result["best_val_loss_epoch"] = (
                position + 1
            )

    if "val_accuracy" in metrics.columns:
        val_accuracy = pd.to_numeric(
            metrics["val_accuracy"],
            errors="coerce",
        )

        if val_accuracy.notna().any():
            values = val_accuracy.to_numpy()

            position = int(
                np.nanargmax(values)
            )

            result["best_val_accuracy"] = float(
                val_accuracy.iloc[position]
            )

            result["best_val_accuracy_epoch"] = (
                position + 1
            )

    return result


def plot_training_history(
    output_dir: Path,
) -> None:
    """
    Generate training loss and accuracy plots.
    """

    metrics_path = (
        output_dir
        / "training_metrics.csv"
    )

    if not metrics_path.is_file():
        return

    import matplotlib.pyplot as plt

    metrics = pd.read_csv(
        metrics_path
    )

    if metrics.empty:
        return

    if "loss" in metrics.columns:
        plt.figure()

        plt.plot(
            metrics["loss"],
            label="train_loss",
        )

        if "val_loss" in metrics.columns:
            plt.plot(
                metrics["val_loss"],
                label="val_loss",
            )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.tight_layout()

        path = (
            output_dir
            / "training_loss.png"
        )

        plt.savefig(
            path,
            dpi=150,
        )

        plt.close()

        print(
            f"Loss plot saved to: {path}"
        )

    if "accuracy" in metrics.columns:
        plt.figure()

        plt.plot(
            metrics["accuracy"],
            label="train_accuracy",
        )

        if "val_accuracy" in metrics.columns:
            plt.plot(
                metrics["val_accuracy"],
                label="val_accuracy",
            )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy")
        plt.legend()
        plt.tight_layout()

        path = (
            output_dir
            / "training_accuracy.png"
        )

        plt.savefig(
            path,
            dpi=150,
        )

        plt.close()

        print(
            f"Accuracy plot saved to: {path}"
        )


def evaluate(
    config: dict[str, Any] | None = None,
    model_path: str | None = None,
) -> EvaluationReport:
    """
    Evaluate a trained model exclusively on the test split.
    """

    if config is None:
        config = load_config()

    output_dir = Path(
        config.get(
            "environment",
            {},
        ).get(
            "output_dir",
            "output",
        )
    )

    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_file = get_model_path(
        config,
        model_path,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("CATTLE MODEL EVALUATION")
    print("=" * 60)

    print(f"Model: {model_file}")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    dataset_info = get_dataset_info(
        config
    )

    class_names = [
        str(class_name)
        for class_name in dataset_info[
            "classes"
        ]
    ]

    if not class_names:
        raise ValueError(
            "No classes were found in the dataset."
        )

    print(
        f"Classes: {class_names}"
    )

    print(
        f"Total images: "
        f"{dataset_info['total_images']}"
    )

    (
        _train_loader,
        _validation_loader,
        test_loader,
        dataset_class_names,
    ) = build_dataloaders(config)

    if class_names != dataset_class_names:
        raise ValueError(
            "Dataset class names are inconsistent."
        )

    model = build_model(config)

    checkpoint = torch.load(
        model_file,
        map_location=device,
        weights_only=True,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise ValueError(
            "The checkpoint does not contain "
            "a model state_dict."
        )

    model.load_state_dict(
        checkpoint
    )

    model.to(device)
    model.eval()

    input_shape = config["model"][
        "input_shape"
    ]

    height = int(
        input_shape[0]
    )

    width = int(
        input_shape[1]
    )

    sample = torch.zeros(
        1,
        3,
        height,
        width,
        device=device,
    )

    with torch.inference_mode():
        output_shape = model(
            sample
        ).shape

    if len(output_shape) != 2:
        raise ValueError(
            "The model does not produce "
            "classification logits."
        )

    model_num_classes = int(
        output_shape[1]
    )

    if model_num_classes != len(
        class_names
    ):
        raise ValueError(
            "Model output classes do not "
            "match the dataset.\n"
            f"Model:   {model_num_classes}\n"
            f"Dataset: {len(class_names)}"
        )

    print("\nGenerating predictions...")

    y_true, y_pred = predict_dataset(
        model,
        test_loader,
        device,
    )

    summary, per_class_df = calculate_metrics(
        y_true,
        y_pred,
        class_names,
    )

    print("\nMetrics")
    print("-------")

    print(
        f"Accuracy:             "
        f"{summary['accuracy']:.4f}"
    )

    print(
        f"Precision (macro):    "
        f"{summary['precision_macro']:.4f}"
    )

    print(
        f"Recall (macro):       "
        f"{summary['recall_macro']:.4f}"
    )

    print(
        f"F1 (macro):           "
        f"{summary['f1_macro']:.4f}"
    )

    print(
        f"Precision (weighted): "
        f"{summary['precision_weighted']:.4f}"
    )

    print(
        f"Recall (weighted):    "
        f"{summary['recall_weighted']:.4f}"
    )

    print(
        f"F1 (weighted):        "
        f"{summary['f1_weighted']:.4f}"
    )

    print("\nPer-class metrics")
    print("-----------------")

    print(
        per_class_df.to_string(
            index=False
        )
    )

    per_class_path = (
        output_dir
        / "per_class_metrics.csv"
    )

    per_class_df.to_csv(
        per_class_path,
        index=False,
    )

    print(
        f"\nPer-class metrics saved to: "
        f"{per_class_path}"
    )

    cm = save_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        output_dir,
    )

    training_history = (
        analyze_training_history(
            output_dir
        )
    )

    plot_training_history(
        output_dir
    )

    per_class_json = {
        str(row["class"]): {
            "precision": float(
                row["precision"]
            ),
            "recall": float(
                row["recall"]
            ),
            "f1_score": float(
                row["f1_score"]
            ),
            "support": int(
                row["support"]
            ),
        }
        for _, row in per_class_df.iterrows()
    }

    report: EvaluationReport = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "model": {
            "path": str(model_file),
            "parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
            ),
        },

        "dataset": {
            "classes": class_names,
            "num_classes": len(
                class_names
            ),
            "total_images": int(
                dataset_info[
                    "total_images"
                ]
            ),
            "class_counts": dataset_info[
                "class_counts"
            ],
        },

        "test": {
            "samples": int(
                len(y_true)
            ),
        },

        "metrics": summary,

        "per_class": per_class_json,

        "confusion_matrix": cm.tolist(),

        "training": training_history,
    }

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    report_path = (
        output_dir
        / f"evaluation_{timestamp}.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nEvaluation report saved to: "
        f"{report_path}"
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the cattle classification "
            "model against the test dataset."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Path to a .pt model checkpoint. "
            "Defaults to best_model.pt or "
            "final_model.pt."
        ),
    )

    args = parser.parse_args()

    evaluate(
        model_path=args.model
    )


if __name__ == "__main__":
    main()
