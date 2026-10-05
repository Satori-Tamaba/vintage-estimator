"""HTTP-слой: принимает фото от пользователя и отдаёт результат пайплайна.

Это точка входа "User -> API" со схемы (imgs/img.png). Внутренние контракты
между модулями описаны в schemas.py и из дataclass-ов не выходят наружу —
здесь они превращаются в обычный JSON через response_model.
"""

import io

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image as PILImage
from pydantic import BaseModel

from vintage_estimator.api.pipeline import estimate

app = FastAPI(title="vintage-estimator")

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
    price: float
    similar_items: list[RetrievedItemResponse]


@app.post("/estimate", response_model=EstimationResponse)
async def estimate_endpoint(file: UploadFile = File(...)) -> EstimationResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"неподдерживаемый тип файла: {file.content_type}",
        )

    data = await file.read()
    if len(data) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"файл больше {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)} МБ",
        )

    try:
        image = np.array(PILImage.open(io.BytesIO(data)))
    except Exception as exc:
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
