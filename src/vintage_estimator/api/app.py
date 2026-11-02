"""HTTP-слой: принимает фото от пользователя и отдаёт результат пайплайна.

Это точка входа "User -> API" со схемы (imgs/img.png). Внутренние контракты
между модулями описаны в schemas.py и из дataclass-ов не выходят наружу —
здесь они превращаются в обычный JSON через response_model.
"""

import io
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image as PILImage, UnidentifiedImageError
from pydantic import BaseModel

from vintage_estimator.api.pipeline import estimate


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    data_dir = os.environ.get("VINTAGE_CATALOG_DIR")
    if data_dir:
        from vintage_estimator.api.runtime import initialize_catalog

        initialize_catalog(data_dir)
    yield


app = FastAPI(title="vintage-estimator", lifespan=lifespan)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024


class RetrievedItemResponse(BaseModel):
    item_id: str
    score: float
    price: float | None = None
    title: str | None = None
    image_url: str | None = None


class EstimationResponse(BaseModel):
    item: str
    style: str | None
    price: float | None
    similar_items: list[RetrievedItemResponse]


@app.post("/estimate", response_model=EstimationResponse)
async def estimate_endpoint(file: UploadFile = File(...)) -> EstimationResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"неподдерживаемый тип файла: {file.content_type}",
        )

    data = await file.read(MAX_IMAGE_SIZE_BYTES + 1)
    if len(data) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"файл больше {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)} МБ",
        )

    try:
        with PILImage.open(io.BytesIO(data)) as uploaded:
            image = np.array(uploaded.convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError, PILImage.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail=f"не удалось прочитать изображение: {exc}") from exc

    result = estimate(image)

    return EstimationResponse(
        item=result.item,
        style=result.style,
        price=result.price,
        similar_items=[
            RetrievedItemResponse(
                item_id=i.item_id,
                score=i.score,
                price=i.price,
                title=i.title,
                image_url=i.image_url,
            )
            for i in result.similar_items
        ],
    )
