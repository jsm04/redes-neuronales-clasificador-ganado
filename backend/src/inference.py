"""Carga del checkpoint entrenado y predicción de razas.

Este módulo es independiente del entrenamiento: no importa el script de
`colab/` ni prepara datasets. Solo necesita el checkpoint y una imagen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

MODEL_NAME = 'efficientnet_b0'


class Inference:
    """Carga el modelo una vez y predice sobre imágenes en memoria o en disco."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.transform = EfficientNet_B0_Weights.DEFAULT.transforms()
        self.class_names: list[str] = []
        self.network: nn.Module | None = None

    def load(self) -> None:
        """Carga la arquitectura y los parámetros guardados."""
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f'No existe el checkpoint: {self.checkpoint_path}'
            )

        try:
            checkpoint: Any = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            # Compatibilidad con versiones de PyTorch sin weights_only.
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
            )

        if not isinstance(checkpoint, dict):
            raise ValueError('El checkpoint debe ser un diccionario.')

        if checkpoint.get('model_name') != MODEL_NAME:
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

        final_layer = network.classifier[1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError('Se esperaba nn.Linear como capa final.')

        network.classifier[1] = nn.Linear(
            in_features=final_layer.in_features,
            out_features=len(class_names),
        )
        network.load_state_dict(state_dict)
        network.to(self.device)
        network.eval()

        self.class_names = class_names
        self.network = network

    def predict(
        self,
        image: Image.Image | str | Path,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Devuelve las clases más probables y sus probabilidades."""
        if top_k < 1:
            raise ValueError('top_k debe ser al menos 1.')

        if self.network is None:
            self.load()

        if self.network is None:
            raise RuntimeError('No se pudo cargar el modelo.')

        rgb = self._to_rgb(image)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            probabilities = torch.softmax(self.network(tensor), dim=1)[0]

        count = min(top_k, len(self.class_names))
        scores, indices = probabilities.topk(count)

        return [
            {
                'breed': self.class_names[index],
                'confidence': round(score, 4),
            }
            for score, index in zip(
                scores.cpu().tolist(),
                indices.cpu().tolist(),
                strict=True,
            )
        ]

    def info(self) -> dict[str, Any]:
        return {
            'model': MODEL_NAME,
            'device': str(self.device),
            'class_names': list(self.class_names),
        }

    def _to_rgb(self, image: Image.Image | str | Path) -> Image.Image:
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.is_file():
                raise FileNotFoundError(f'No se encontró la imagen: {path}')
            with Image.open(path) as opened:
                return opened.convert('RGB')

        return image.convert('RGB')
