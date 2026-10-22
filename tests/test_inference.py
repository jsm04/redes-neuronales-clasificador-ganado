"""Verifica que el checkpoint real se carga y que la API responde."""

from __future__ import annotations

from pathlib import Path

import pytest
from cattle_api.inference import Inference
from cattle_api.main import app
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    REPO_ROOT / 'data' / 'NN' / 'cattle_training' / 'best_cattle_breed_model.pth'
)
IMAGE = (
    REPO_ROOT
    / 'data'
    / 'NN'
    / 'cattle_dataset'
    / 'jersey-cattle'
    / 'Jerseycattle10.jpg'
)

pytestmark = pytest.mark.skipif(
    not CHECKPOINT.is_file() or not IMAGE.is_file(),
    reason='Faltan el checkpoint o una imagen del conjunto de datos.',
)


def test_inference_predicts_known_breed() -> None:
    engine = Inference(CHECKPOINT, 'cpu')
    engine.load()

    assert engine.info()['class_names'], 'El checkpoint no expone clases.'
    predictions = engine.predict(IMAGE, top_k=3)

    assert len(predictions) == 3
    assert predictions[0]['breed'] == 'jersey-cattle'
    assert sum(item['confidence'] for item in predictions) <= 1


def test_api_predicts_from_upload() -> None:
    with TestClient(app) as client:
        response = client.post(
            '/api/v1/predictions',
            files={'file': ('cow.jpg', IMAGE.read_bytes(), 'image/jpeg')},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload['model'] == 'efficientnet_b0'
        assert payload['predictions'][0]['breed'] == 'jersey-cattle'


def test_api_rejects_non_image_upload() -> None:
    with TestClient(app) as client:
        response = client.post(
            '/api/v1/predictions',
            files={'file': ('notes.txt', b'hola', 'text/plain')},
        )

        assert response.status_code == 415
