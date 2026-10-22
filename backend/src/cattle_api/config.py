"""Configuración del servicio de inferencia.

Las rutas se resuelven desde la ubicación de este archivo, nunca desde el
directorio de trabajo del shell.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

CHECKPOINT_PATH = Path(
    os.getenv(
        'CATTLE_CHECKPOINT',
        REPO_ROOT / 'data' / 'NN' / 'cattle_training' / 'best_cattle_breed_model.pth',
    )
).resolve()

# None deja que PyTorch elija CUDA si está disponible.
DEVICE = os.getenv('CATTLE_DEVICE') or None

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        'CATTLE_CORS_ORIGINS',
        'http://localhost:3000,http://127.0.0.1:3000',
    ).split(',')
    if origin.strip()
]

MAX_UPLOAD_BYTES = int(os.getenv('CATTLE_MAX_UPLOAD_BYTES', 10 * 1024 * 1024))
TOP_K = int(os.getenv('CATTLE_TOP_K', 3))
