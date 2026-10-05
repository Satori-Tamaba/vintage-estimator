"""HTTP-слой: принимает фото от пользователя и отдаёт результат пайплайна.

Это точка входа "User -> API" со схемы (imgs/img.png). Внутренние контракты
между модулями описаны в schemas.py и из дataclass-ов не выходят наружу —
здесь они превращаются в обычный JSON через response_model.
"""

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image as PILImage
from pydantic import BaseModel

from vintage_estimator.api.pipeline import estimate

app = FastAPI(title="vintage-estimator")


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
    try:
        image = np.array(PILImage.open(file.file))
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
