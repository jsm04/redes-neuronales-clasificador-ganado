"""Entrenamiento y evaluación de una red neuronal para clasificar razas bovinas.

Diseñado para Google Colab, pero organizado en clases para facilitar su análisis,
mantenimiento y ejecución desde VS Code.

Flujo: Data (imágenes → tensores) → Model (EfficientNet-B0) → Trainer
(entrena y guarda el checkpoint) → Evaluation (métricas) → Inference
(carga el checkpoint y predice).
"""

from __future__ import annotations

import copy
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


@dataclass(frozen=True)
class Config:
    """Configuración central del experimento."""

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    dataset_path: Path = PROJECT_ROOT / 'data' / 'NN' / 'cattle_dataset'
    output_dir: Path = PROJECT_ROOT / 'data' / 'NN' / 'cattle_training'
    seed: int = 42
    # 224x224 es la resolución con la que se preentrenó EfficientNet-B0.
    image_size: int = 224
    # Imágenes por paso; valores mayores usan más memoria de GPU.
    batch_size: int = 32
    num_workers: int = 2
    # Épocas con el backbone congelado antes de iniciar el ajuste fino.
    head_epochs: int = 3
    # Tope total de épocas y paciencia del early stopping.
    max_epochs: int = 20
    patience: int = 5
    weight_decay: float = 1e-4
    # LR alto para la cabeza nueva; LR bajo para no destruir el backbone.
    head_learning_rate: float = 1e-3
    backbone_learning_rate: float = 1e-4
    classifier_learning_rate: float = 3e-4
    minimum_images_per_class: int = 10
    # Precisión mixta: entrena en float16 en GPU y así ocupa menos memoria.
    use_amp: bool = True


@dataclass
class EpochMetrics:
    """Métricas de una época de entrenamiento o evaluación."""

    loss: float
    accuracy: float
    macro_f1: float


@dataclass
class TrainingResult:
    """Resumen del entrenamiento."""

    best_epoch: int
    best_validation_macro_f1: float
    history: list[dict[str, float | int]]
    elapsed_seconds: float
    checkpoint_path: Path


class Reproducibility:
    """Configura las semillas para reducir la variación entre ejecuciones."""

    @staticmethod
    def configure(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Estas opciones priorizan resultados reproducibles cuando es posible.
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class Runtime:
    """Gestiona el dispositivo de cómputo y la precisión mixta."""

    def __init__(self, use_amp: bool = True) -> None:
        # Usa la GPU si existe; en CPU el entrenamiento será mucho más lento.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.amp_enabled = use_amp and self.device.type == 'cuda'
        self.amp_dtype = torch.float16
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.amp_enabled,
        )

    def autocast(self):
        """Devuelve el contexto de precisión mixta apropiado para el dispositivo."""
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        )


class Data:
    """Descubre las imágenes, divide el conjunto y crea los DataLoaders."""

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    class ImageSubset(Dataset[tuple[torch.Tensor, int]]):
        """Subconjunto de imágenes con transformaciones específicas."""

        def __init__(
            self,
            samples: list[tuple[str, int]],
            indices: np.ndarray,
            transform: Any,
        ) -> None:
            self.samples = samples
            self.indices = [int(index) for index in indices]
            self.transform = transform

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            path, label = self.samples[self.indices[index]]

            # El archivo se cierra automáticamente al salir del bloque.
            with Image.open(path) as image:
                tensor = self.transform(image.convert('RGB'))

            return tensor, label

    class ImageSubsetWithPaths(Dataset[tuple[torch.Tensor, int, str]]):
        """Subconjunto de evaluación que también devuelve la ruta del archivo."""

        def __init__(
            self,
            samples: list[tuple[str, int]],
            indices: np.ndarray,
            transform: Any,
        ) -> None:
            self.samples = samples
            self.indices = [int(index) for index in indices]
            self.transform = transform

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
            path, label = self.samples[self.indices[index]]

            with Image.open(path) as image:
                tensor = self.transform(image.convert('RGB'))

            return tensor, label, path

    def __init__(self, config: Config, weights: EfficientNet_B0_Weights) -> None:
        self.config = config
        self.root = config.dataset_path
        self.weights = weights

        self.dataset: ImageFolder | None = None
        self.class_names: list[str] = []
        self.train_indices = np.array([], dtype=np.int64)
        self.val_indices = np.array([], dtype=np.int64)
        self.test_indices = np.array([], dtype=np.int64)
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

        # Aumentos: el modelo ve variantes de la misma foto y generaliza mejor.
        # Recorte aleatorio al 75-100% del área, espejo, giro y cambios de color.
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    config.image_size,
                    scale=(0.75, 1.0),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(12),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                ),
                # PIL → tensor float32 en [0, 1] con forma (3, 224, 224).
                transforms.ToTensor(),
                # Normaliza con la media y desviación de ImageNet, la misma
                # escala con la que se preentrenó el backbone.
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        # En validación y prueba no se aumenta: solo recorte central + normalización.
        self.eval_transform = weights.transforms()

    def prepare(self) -> None:
        """Valida el conjunto de imágenes, divide los datos y crea los cargadores."""
        if not self.root.is_dir():
            raise FileNotFoundError(f'No se encontró el conjunto de datos: {self.root}')

        dataset = ImageFolder(self.root)
        if not dataset.samples:
            raise ValueError(f'No se encontraron imágenes en: {self.root}')

        self.dataset = dataset
        self.class_names = list(dataset.classes)

        if len(self.class_names) < 2:
            raise ValueError('Se necesitan al menos dos carpetas de razas.')

        counts = Counter(dataset.targets)
        print(f'Razas: {len(self.class_names)}')
        print(f'Imágenes: {len(dataset)}')
        for index, name in enumerate(self.class_names):
            print(f'{name:25} {counts[index]:5} imágenes')

        if min(counts.values()) < self.config.minimum_images_per_class:
            raise ValueError(
                'Cada raza debe tener al menos '
                f'{self.config.minimum_images_per_class} imágenes.'
            )

        # División estratificada 80/10/10: cada parte conserva la proporción
        # de razas, importante cuando algunas clases tienen menos fotos.
        all_indices = np.arange(len(dataset), dtype=np.int64)
        all_labels = np.asarray(dataset.targets, dtype=np.int64)

        self.train_indices, temporary_indices = train_test_split(
            all_indices,
            test_size=0.20,
            random_state=self.config.seed,
            stratify=all_labels,
        )
        self.val_indices, self.test_indices = train_test_split(
            temporary_indices,
            test_size=0.50,
            random_state=self.config.seed,
            stratify=all_labels[temporary_indices],
        )

        print(
            f'Entrenamiento: {len(self.train_indices)} imágenes '
            f'({len(self.train_indices) / len(dataset):.0%})'
        )
        print(
            f'Validación:    {len(self.val_indices)} imágenes '
            f'({len(self.val_indices) / len(dataset):.0%})'
        )
        print(
            f'Prueba:        {len(self.test_indices)} imágenes '
            f'({len(self.test_indices) / len(dataset):.0%})'
        )

        # En CPU, pin_memory no aporta la misma ventaja que en CUDA.
        pin_memory = torch.cuda.is_available()
        workers = self.config.num_workers
        persistent_workers = workers > 0

        self.train_loader = DataLoader(
            self.ImageSubset(dataset.samples, self.train_indices, self.train_transform),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        self.val_loader = DataLoader(
            self.ImageSubset(dataset.samples, self.val_indices, self.eval_transform),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        self.test_loader = DataLoader(
            self.ImageSubset(dataset.samples, self.test_indices, self.eval_transform),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    def test_loader_with_paths(self) -> DataLoader:
        """Crea un cargador de prueba que conserva las rutas originales."""
        if self.dataset is None:
            raise RuntimeError('Primero debe prepararse el conjunto de datos.')

        return DataLoader(
            self.ImageSubsetWithPaths(
                self.dataset.samples,
                self.test_indices,
                self.eval_transform,
            ),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.config.num_workers > 0,
        )

    def training_class_weights(self, device: torch.device) -> torch.Tensor:
        """Calcula pesos inversamente proporcionales a las frecuencias de entrenamiento."""
        if self.dataset is None:
            raise RuntimeError('Primero debe prepararse el conjunto de datos.')

        # Razas con menos fotos reciben un peso mayor en la pérdida.
        labels = np.asarray(self.dataset.targets, dtype=np.int64)[self.train_indices]
        counts = np.bincount(labels, minlength=len(self.class_names))
        weights = len(labels) / (len(self.class_names) * np.maximum(counts, 1))
        return torch.tensor(weights, dtype=torch.float32, device=device)


class Model:
    """Construye y configura EfficientNet-B0 para las razas del conjunto."""

    NAME = 'efficientnet_b0'

    def __init__(
        self,
        class_count: int,
        weights: EfficientNet_B0_Weights,
        device: torch.device,
    ) -> None:
        self.device = device
        # EfficientNet-B0 es una CNN preentrenada en ImageNet (~5.3 M de
        # parámetros con 1000 clases). Recibe imágenes 224x224x3 y devuelve
        # logits; la parte convolucional actúa como extractor de características.
        self.network = efficientnet_b0(weights=weights)

        classifier = self.network.classifier[1]
        if not isinstance(classifier, nn.Linear):
            raise TypeError('Se esperaba una capa nn.Linear en el clasificador.')

        # Se sustituye la capa final de 1000 clases: 1280 características de
        # entrada por una neurona de salida por raza. Con 5 razas son 6405
        # parámetros nuevos; el resto del modelo (~4.0 M) se reutiliza.
        self.network.classifier[1] = nn.Linear(
            in_features=classifier.in_features,
            out_features=class_count,
        )
        self.network = self.network.to(device)

    def freeze_backbone(self) -> None:
        """Congela las capas convolucionales para entrenar primero el clasificador."""
        # requires_grad=False evita que el backbone cambie: solo aprende la cabeza.
        for parameter in self.network.features.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Habilita todas las capas para el ajuste fino."""
        # Se descongela todo, pero el backbone usa un learning rate más bajo.
        for parameter in self.network.features.parameters():
            parameter.requires_grad = True

    def backbone_is_frozen(self) -> bool:
        """Indica si todos los parámetros del backbone están congelados."""
        return all(
            not parameter.requires_grad
            for parameter in self.network.features.parameters()
        )


class Trainer:
    """Ejecuta el entrenamiento, la validación y el guardado del mejor modelo."""

    def __init__(
        self,
        config: Config,
        data: Data,
        model: Model,
        runtime: Runtime,
        criterion: nn.Module,
    ) -> None:
        self.config = config
        self.data = data
        self.model = model
        self.runtime = runtime
        self.criterion = criterion
        self.checkpoint_path = config.output_dir / 'best_cattle_breed_model.pth'

        # Fase 1: solo se optimiza el clasificador nuevo; el backbone está congelado.
        self.optimizer = torch.optim.AdamW(
            model.network.classifier.parameters(),
            lr=config.head_learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = self._make_scheduler(self.optimizer)

    @staticmethod
    def _make_scheduler(
        optimizer: torch.optim.Optimizer | None = None,
    ) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        # Se crea nuevamente al cambiar de entrenamiento de cabeza a ajuste fino.
        if optimizer is None:
            raise ValueError('Se necesita un optimizador para crear el scheduler.')
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.3,
            patience=2,
        )

    def _configure_fine_tuning(self) -> None:
        self.model.unfreeze_backbone()
        # Fase 2: dos grupos con LR distinto; AdamW actualiza cada grupo con el suyo.
        self.optimizer = torch.optim.AdamW(
            [
                {
                    'params': self.model.network.features.parameters(),
                    'lr': self.config.backbone_learning_rate,
                },
                {
                    'params': self.model.network.classifier.parameters(),
                    'lr': self.config.classifier_learning_rate,
                },
            ],
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = self._make_scheduler(self.optimizer)
        print('\nSe habilitó el extractor de características para ajuste fino.')

    def run_epoch(self, loader: DataLoader, training: bool) -> EpochMetrics:
        """Ejecuta una época y calcula pérdida, exactitud y F1 macro."""
        network = self.model.network
        network.train(training)

        # BatchNorm y otras capas del extractor permanecen estables mientras está congelado.
        if training and self.model.backbone_is_frozen():
            network.features.eval()

        total_loss = 0.0
        predictions: list[int] = []
        targets: list[int] = []

        # Cada lote es (images, labels): float32 (float16 con autocast) de forma
        # (lote, 3, 224, 224) y etiquetas int64 de forma (lote,).
        for images, labels, *_ in loader:
            images = images.to(self.runtime.device, non_blocking=True)
            labels = labels.to(self.runtime.device, non_blocking=True)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                with self.runtime.autocast():
                    # Logits de forma (lote, número de razas), sin softmax todavía.
                    logits = network(images)
                    loss = self.criterion(logits, labels)

                if training:
                    # GradScaler evita que los gradientes float16 se desborden.
                    self.runtime.scaler.scale(loss).backward()
                    self.runtime.scaler.step(self.optimizer)
                    self.runtime.scaler.update()

            # Se acumula la pérdida y se multiplica por el tamaño del lote.
            total_loss += loss.item() * labels.size(0)
            # argmax elige la clase con mayor logit; se compara con la etiqueta real.
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            targets.extend(labels.detach().cpu().tolist())

        if not targets:
            raise ValueError('El cargador de datos está vacío.')

        # F1 macro promedia las razas por igual: no premia a las mayoritarias.
        return EpochMetrics(
            loss=total_loss / len(targets),
            accuracy=float(np.mean(np.asarray(predictions) == np.asarray(targets))),
            macro_f1=float(
                f1_score(
                    targets,
                    predictions,
                    average='macro',
                    zero_division=0,
                )
            ),
        )

    def fit(self) -> TrainingResult:
        """Entrena hasta alcanzar el máximo de épocas o activar early stopping."""
        if self.data.train_loader is None or self.data.val_loader is None:
            raise RuntimeError('Primero debe prepararse el conjunto de datos.')

        best_f1 = -1.0
        best_epoch = 0
        epochs_without_improvement = 0
        history: list[dict[str, float | int]] = []
        start_time = time.time()

        for epoch in range(self.config.max_epochs):
            # Primero la cabeza nueva con backbone congelado; luego fine-tuning.
            if epoch == self.config.head_epochs:
                self._configure_fine_tuning()

            train_metrics = self.run_epoch(self.data.train_loader, training=True)
            val_metrics = self.run_epoch(self.data.val_loader, training=False)
            self.scheduler.step(val_metrics.macro_f1)

            record: dict[str, float | int] = {
                'epoch': epoch + 1,
                'train_loss': train_metrics.loss,
                'train_accuracy': train_metrics.accuracy,
                'train_macro_f1': train_metrics.macro_f1,
                'val_loss': val_metrics.loss,
                'val_accuracy': val_metrics.accuracy,
                'val_macro_f1': val_metrics.macro_f1,
            }
            history.append(record)

            print(
                f'Época {epoch + 1:02}/{self.config.max_epochs} | '
                f'entrenamiento: pérdida {train_metrics.loss:.4f}, '
                f'exactitud {train_metrics.accuracy:.3f} | '
                f'validación: pérdida {val_metrics.loss:.4f}, '
                f'exactitud {val_metrics.accuracy:.3f}, '
                f'F1 macro {val_metrics.macro_f1:.3f}'
            )

            if val_metrics.macro_f1 > best_f1:
                best_f1 = val_metrics.macro_f1
                best_epoch = epoch + 1
                epochs_without_improvement = 0

                # Se guardan tensores independientes para no depender del estado posterior.
                torch.save(
                    {
                        'model_state_dict': copy.deepcopy(
                            self.model.network.state_dict()
                        ),
                        'class_names': self.data.class_names,
                        'model_name': Model.NAME,
                        'image_size': self.config.image_size,
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225],
                        'best_epoch': best_epoch,
                        'val_macro_f1': best_f1,
                        'seed': self.config.seed,
                    },
                    self.checkpoint_path,
                )
            else:
                epochs_without_improvement += 1

            # Early stopping: se detiene si el F1 de validación deja de mejorar.
            if epochs_without_improvement >= self.config.patience:
                print('Parada temprana: el F1 macro de validación no mejoró.')
                break

        elapsed = time.time() - start_time
        print(f'\nMejor época: {best_epoch}')
        print(f'Mejor F1 macro de validación: {best_f1:.4f}')
        print(f'Tiempo transcurrido: {elapsed / 60:.1f} minutos')
        print(f'Checkpoint: {self.checkpoint_path}')

        return TrainingResult(
            best_epoch=best_epoch,
            best_validation_macro_f1=best_f1,
            history=history,
            elapsed_seconds=elapsed,
            checkpoint_path=self.checkpoint_path,
        )


class Evaluation:
    """Evalúa el checkpoint y genera métricas, matriz y análisis de errores."""

    def __init__(
        self,
        config: Config,
        data: Data,
        model: Model,
        trainer: Trainer,
        runtime: Runtime,
    ) -> None:
        self.config = config
        self.data = data
        self.model = model
        self.trainer = trainer
        self.runtime = runtime

    def load_best_checkpoint(self) -> dict[str, Any]:
        """Carga el checkpoint local, priorizando la compatibilidad con pesos guardados."""
        if not self.trainer.checkpoint_path.is_file():
            raise FileNotFoundError(
                f'No existe el checkpoint: {self.trainer.checkpoint_path}'
            )

        # El checkpoint contiene tensores y metadatos propios del entrenamiento.
        try:
            checkpoint = torch.load(
                self.trainer.checkpoint_path,
                map_location=self.runtime.device,
                weights_only=True,
            )
        except TypeError:
            # Compatibilidad con versiones de PyTorch que no exponen weights_only.
            checkpoint = torch.load(
                self.trainer.checkpoint_path,
                map_location=self.runtime.device,
            )

        self.model.network.load_state_dict(checkpoint['model_state_dict'])
        self.model.network.eval()
        return checkpoint

    def run(self) -> dict[str, Any]:
        """Calcula métricas del conjunto de prueba y guarda los resultados."""
        if self.data.test_loader is None:
            raise RuntimeError('Primero debe prepararse el conjunto de datos.')

        checkpoint = self.load_best_checkpoint()
        test_metrics = self.trainer.run_epoch(self.data.test_loader, training=False)

        print(f'Pérdida de prueba: {test_metrics.loss:.4f}')
        print(f'Exactitud de prueba: {test_metrics.accuracy:.4f}')
        print(f'F1 macro de prueba: {test_metrics.macro_f1:.4f}')

        y_true: list[int] = []
        y_pred: list[int] = []

        with torch.inference_mode():
            for images, labels in self.data.test_loader:
                images = images.to(self.runtime.device, non_blocking=True)
                with self.runtime.autocast():
                    logits = self.model.network(images)
                y_pred.extend(logits.argmax(dim=1).cpu().tolist())
                y_true.extend(labels.tolist())

        # Reporte por raza: precisión, recall y F1 de cada clase.
        print('\nMétricas por raza:\n')
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(self.data.class_names))),
            target_names=self.data.class_names,
            digits=3,
            zero_division=0,
        )
        print(report)

        # Matriz de confusión: filas = raza real, columnas = raza predicha.
        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(self.data.class_names))),
        )
        figure, axis = plt.subplots(
            figsize=(
                max(8, len(self.data.class_names) * 0.7),
                max(6, len(self.data.class_names) * 0.6),
            )
        )
        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=self.data.class_names,
        ).plot(ax=axis, xticks_rotation=90, cmap='Blues', colorbar=False)
        axis.set_title('Clasificación de razas bovinas — conjunto de prueba')
        figure.tight_layout()
        figure.savefig(self.config.output_dir / 'confusion_matrix.png', dpi=150)
        plt.show()
        plt.close(figure)

        errors = self.find_misclassifications()
        self.plot_misclassifications(errors)

        results: dict[str, Any] = {
            'model': Model.NAME,
            'best_epoch': int(checkpoint['best_epoch']),
            'best_validation_macro_f1': float(checkpoint['val_macro_f1']),
            'test_loss': float(test_metrics.loss),
            'test_accuracy': float(test_metrics.accuracy),
            'test_macro_f1': float(test_metrics.macro_f1),
            'class_names': self.data.class_names,
            'train_images': len(self.data.train_indices),
            'validation_images': len(self.data.val_indices),
            'test_images': len(self.data.test_indices),
            'seed': self.config.seed,
            'device': str(self.runtime.device),
            'misclassified_images': len(errors),
        }

        with (self.config.output_dir / 'evaluation.json').open(
            'w',
            encoding='utf-8',
        ) as file:
            json.dump(results, file, indent=2, ensure_ascii=False)

        pd.DataFrame(self.trainer_history).to_csv(
            self.config.output_dir / 'training_history.csv',
            index=False,
        )
        print(f'Resultados guardados en: {self.config.output_dir}')
        return results

    @property
    def trainer_history(self) -> list[dict[str, float | int]]:
        # El historial se asigna desde la función principal después del entrenamiento.
        return getattr(self, '_history', [])

    @trainer_history.setter
    def trainer_history(self, value: list[dict[str, float | int]]) -> None:
        self._history = value

    def find_misclassifications(self) -> list[dict[str, Any]]:
        """Identifica imágenes de prueba cuya clase predicha no coincide con la real."""
        loader = self.data.test_loader_with_paths()
        errors: list[dict[str, Any]] = []
        self.model.network.eval()

        with torch.inference_mode():
            for images, labels, paths in loader:
                images = images.to(self.runtime.device, non_blocking=True)
                with self.runtime.autocast():
                    logits = self.model.network(images)

                probabilities = torch.softmax(logits, dim=1).cpu()
                predictions = logits.argmax(dim=1).cpu()

                for index, true_label in enumerate(labels.tolist()):
                    predicted_label = int(predictions[index].item())
                    if predicted_label != true_label:
                        errors.append(
                            {
                                'path': paths[index],
                                'true': self.data.class_names[true_label],
                                'predicted': self.data.class_names[predicted_label],
                                'confidence': float(
                                    probabilities[index, predicted_label].item()
                                ),
                            }
                        )

        with (self.config.output_dir / 'misclassifications.json').open(
            'w',
            encoding='utf-8',
        ) as file:
            json.dump(errors, file, indent=2, ensure_ascii=False)

        print(f'Total de imágenes mal clasificadas: {len(errors)}')
        for error in errors:
            print(error['path'])

        return errors

    def plot_misclassifications(self, errors: list[dict[str, Any]]) -> None:
        """Muestra hasta ocho errores sin fallar si el modelo no cometió ninguno."""
        if not errors:
            print(
                'No se encontraron errores de clasificación en el conjunto de prueba.'
            )
            return

        visible_errors = errors[:8]
        columns = 2
        rows = (len(visible_errors) + columns - 1) // columns
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(12, 4 * rows),
            squeeze=False,
        )

        for axis, error in zip(axes.flat, visible_errors, strict=False):
            with Image.open(error['path']) as image:
                axis.imshow(image.convert('RGB'))
            axis.set_title(
                f'Real: {error["true"]}\n'
                f'Predicción: {error["predicted"]} '
                f'({error["confidence"]:.1%})'
            )
            axis.axis('off')

        for axis in axes.flat[len(visible_errors) :]:
            axis.axis('off')

        figure.tight_layout()
        figure.savefig(
            self.config.output_dir / 'misclassified_images.png',
            dpi=150,
        )
        plt.show()
        plt.close(figure)


class Inference:
    """Carga el modelo entrenado y predice las razas más probables."""

    def __init__(
        self,
        checkpoint_path: Path,
        runtime: Runtime,
        weights: EfficientNet_B0_Weights,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.runtime = runtime
        self.transform = weights.transforms()
        self.checkpoint: dict[str, Any] | None = None
        self.network: nn.Module | None = None

    def load(self) -> None:
        """Carga la arquitectura y los parámetros guardados."""
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f'No existe el checkpoint: {self.checkpoint_path}')

        checkpoint: Any = torch.load(
            self.checkpoint_path,
            map_location=self.runtime.device,
            weights_only=True,
        )

        if not isinstance(checkpoint, dict):
            raise ValueError('El checkpoint debe ser un diccionario.')

        if checkpoint.get('model_name') != Model.NAME:
            raise ValueError('La arquitectura del checkpoint no es compatible.')

        class_names = checkpoint.get('class_names')
        state_dict = checkpoint.get('model_state_dict')

        if (
            not isinstance(class_names, list)
            or not class_names
            or not all(isinstance(name, str) for name in class_names)
        ):
            raise ValueError('El checkpoint contiene nombres de clases inválidos.')

        if not isinstance(state_dict, dict):
            raise ValueError('El checkpoint no contiene un estado del modelo válido.')

        network = efficientnet_b0(weights=None)

        classifier = network.classifier
        if not isinstance(classifier, nn.Sequential):
            raise TypeError('Se esperaba nn.Sequential como clasificador.')

        final_layer = classifier[1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError('Se esperaba nn.Linear como capa final.')

        network.classifier[1] = nn.Linear(
            in_features=final_layer.in_features,
            out_features=len(class_names),
        )
        network.load_state_dict(state_dict)
        network.to(self.runtime.device)
        network.eval()

        self.checkpoint = {
            'model_name': Model.NAME,
            'class_names': class_names,
            'model_state_dict': state_dict,
        }
        self.network = network

    def predict(
        self,
        image_path: str | Path,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Devuelve las clases más probables y sus probabilidades."""
        if top_k < 1:
            raise ValueError('top_k debe ser al menos 1.')

        if self.network is None or self.checkpoint is None:
            self.load()

        if self.network is None or self.checkpoint is None:
            raise RuntimeError('No se pudo cargar el modelo.')

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f'No se encontró la imagen: {path}')

        with Image.open(path) as image:
            # unsqueeze(0) crea el lote de una sola imagen: (1, 3, 224, 224).
            tensor = self.transform(image.convert('RGB')).unsqueeze(0)

        tensor = tensor.to(self.runtime.device)

        # softmax convierte los logits en probabilidades que suman 1.
        with torch.inference_mode(), self.runtime.autocast():
            probabilities = torch.softmax(self.network(tensor), dim=1)[0]

        class_names = self.checkpoint['class_names']
        count = min(top_k, len(class_names))
        # topk devuelve las k razas más probables, ya ordenadas de mayor a menor.
        scores, indices = probabilities.topk(count)

        return [
            {
                'breed': class_names[index],
                'confidence': round(score, 4),
            }
            for score, index in zip(
                scores.cpu().tolist(),
                indices.cpu().tolist(),
                strict=True,
            )
        ]


def mount_google_drive() -> None:
    """Monta Google Drive únicamente cuando se ejecuta dentro de Colab."""
    try:
        from google.colab import drive  # pyright: ignore[reportMissingImports]
    except ImportError:
        # Fuera de Colab, las rutas deben apuntar a directorios locales accesibles.
        return

    drive.mount('/content/drive')


def main() -> None:
    """Orquesta la preparación, el entrenamiento, la evaluación y la predicción."""
    config = Config()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    mount_google_drive()
    Reproducibility.configure(config.seed)

    runtime = Runtime(use_amp=config.use_amp)
    print(f'Dispositivo: {runtime.device}')
    if runtime.device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Conjunto de datos: {config.dataset_path}')
    print(f'Directorio de salida: {config.output_dir}')

    weights = EfficientNet_B0_Weights.DEFAULT
    data = Data(config, weights)
    data.prepare()

    model = Model(len(data.class_names), weights, runtime.device)
    model.freeze_backbone()

    criterion = nn.CrossEntropyLoss(
        weight=data.training_class_weights(runtime.device),
    )
    trainer = Trainer(config, data, model, runtime, criterion)

    print(f'Modelo: {Model.NAME}')
    print(f'Número de clases: {len(data.class_names)}')

    training_result = trainer.fit()

    evaluation = Evaluation(config, data, model, trainer, runtime)
    evaluation.trainer_history = training_result.history
    evaluation.run()

    # Cambiar esta ruta por una imagen que se quiera clasificar.
    prediction_image = config.dataset_path.parent / 'cattle_test.jpg'
    if prediction_image.is_file():
        inference = Inference(training_result.checkpoint_path, runtime, weights)
        for prediction in inference.predict(prediction_image):
            print(f'{prediction["breed"]}: {prediction["confidence"]:.1%}')
    else:
        print(f'Predicción de ejemplo omitida: no existe el archivo {prediction_image}')


if __name__ == '__main__':
    main()
