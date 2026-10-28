"""Servicio HTTP que expone el modelo de clasificación de razas bovinas."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from . import config
from .inference import MODEL_NAME, Inference
from .schemas import HealthResponse, Prediction, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    engine = Inference(config.CHECKPOINT_PATH, config.DEVICE)
    engine.load()
    app.state.engine = engine
    yield


app = FastAPI(
    title='Cattle Breed API',
    version='0.1.0',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=['GET', 'POST'],
    allow_headers=['*'],
)


@app.get('/health', response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    engine: Inference = request.app.state.engine
    return HealthResponse(status='ok', **engine.info())


@app.post('/api/v1/predictions', response_model=PredictionResponse)
async def predict(
    request: Request, file: Annotated[UploadFile, File()]
) -> PredictionResponse:
    if not (file.content_type or '').startswith('image/'):
        raise HTTPException(status_code=415, detail='Se esperaba una imagen.')

    data = await file.read(config.MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail='El archivo está vacío.')
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail='La imagen es demasiado grande.')

    try:
        image = Image.open(BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(
            status_code=400,
            detail='No se pudo decodificar la imagen.',
        ) from error

    engine: Inference = request.app.state.engine
    try:
        results = await run_in_threadpool(engine.predict, image, config.TOP_K)
    finally:
        image.close()

    return PredictionResponse(
        model=MODEL_NAME,
        predictions=[Prediction(**item) for item in results],
    )
