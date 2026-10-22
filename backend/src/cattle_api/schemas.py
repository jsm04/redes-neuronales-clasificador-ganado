"""Esquemas de entrada y salida de la API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    breed: str = Field(description='Nombre de la raza en el checkpoint.')
    confidence: float = Field(ge=0, le=1, description='Probabilidad de la raza.')


class PredictionResponse(BaseModel):
    model: str
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    class_names: list[str]
