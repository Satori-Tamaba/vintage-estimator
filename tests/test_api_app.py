"""Проверяет HTTP-слой отдельно от нереализованных стадий пайплайна.

Стадии подменяются заглушками (как в test_pipeline_contract.py), чтобы
тест проверял именно API: приём файла, вызов estimate(), сериализацию ответа.
"""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from vintage_estimator.api import pipeline
from vintage_estimator.api.app import app
from vintage_estimator.schemas import ClassificationResult, PricePrediction, RetrievedItem

client = TestClient(app)


@pytest.mark.parametrize("top_k", [0, -1, 101, "abc"])
def test_endpoint_rejects_invalid_top_k(top_k):
    response = client.post(
        f"/estimate?top_k={top_k}",
        files={"file": ("photo.png", _fake_image_bytes(), "image/png")},
    )
    assert response.status_code == 422


def test_endpoint_runs_estimate_outside_event_loop(monkeypatch):
    import asyncio
    import threading
    from fastapi import UploadFile
    from starlette.datastructures import Headers
    from vintage_estimator.api import app as app_module
    from vintage_estimator.schemas import EstimationResult

    calls = []

    def estimate(image, top_k):
        calls.append((threading.get_ident(), top_k))
        return EstimationResult(item="vase", style=None, price=None)

    monkeypatch.setattr(app_module, "estimate", estimate)
    upload = UploadFile(
        file=io.BytesIO(_fake_image_bytes()),
        headers=Headers({"content-type": "image/png"}),
    )
    asyncio.run(app_module.estimate_endpoint(upload, top_k=12))
    assert calls[0][0] != threading.get_ident()
    assert calls[0][1] == 12


def _fake_image_bytes() -> bytes:
    buf = io.BytesIO()
    PILImage.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def test_estimate_endpoint_returns_pipeline_result(monkeypatch):
    classification = ClassificationResult(category="vase", style="art-deco", confidence=0.9)
    similar_items = [RetrievedItem(item_id="1", score=0.95, price=1000.0, title="Ваза")]
    prediction = PricePrediction(price=1200.0)

    monkeypatch.setattr(pipeline, "preprocess", lambda img: img)
    monkeypatch.setattr(pipeline, "encode", lambda img: np.zeros(8, dtype=np.float32))
    monkeypatch.setattr(pipeline, "classify", lambda emb: classification)
    monkeypatch.setattr(pipeline, "retrieve", lambda emb, top_k=5: similar_items)
    monkeypatch.setattr(pipeline, "predict_price", lambda cls, items: prediction)

    response = client.post(
        "/estimate",
        files={"file": ("photo.png", _fake_image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["item"] == "vase"
    assert body["style"] == "art-deco"
    assert body["price"] == 1200.0
    assert body["similar_items"][0]["item_id"] == "1"


def test_estimate_endpoint_rejects_unsupported_content_type():
    response = client.post(
        "/estimate",
        files={"file": ("not_an_image.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 415


def test_estimate_endpoint_rejects_corrupted_image():
    response = client.post(
        "/estimate",
        files={"file": ("broken.png", b"not actually a png", "image/png")},
    )

    assert response.status_code == 400


def test_estimate_endpoint_rejects_too_large_file():
    oversized = b"0" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/estimate",
        files={"file": ("huge.png", oversized, "image/png")},
    )

    assert response.status_code == 413


def test_estimate_endpoint_serializes_missing_price_as_null(monkeypatch):
    from vintage_estimator.schemas import EstimationResult
    from vintage_estimator.api import app as app_module

    monkeypatch.setattr(app_module, "estimate", lambda image, top_k=5: EstimationResult(
        item="vase", style=None, price=None,
    ))
    response = client.post(
        "/estimate", files={"file": ("photo.png", _fake_image_bytes(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["price"] is None


def test_endpoint_reads_at_most_limit_plus_one_byte():
    import asyncio
    from unittest.mock import AsyncMock

    import pytest
    from fastapi import HTTPException, UploadFile
    from starlette.datastructures import Headers
    from vintage_estimator.api.app import MAX_IMAGE_SIZE_BYTES, estimate_endpoint

    upload = UploadFile(
        filename="huge.png", file=io.BytesIO(),
        headers=Headers({"content-type": "image/png"}),
    )
    upload.read = AsyncMock(return_value=b"0" * (MAX_IMAGE_SIZE_BYTES + 1))
    with pytest.raises(HTTPException) as error:
        asyncio.run(estimate_endpoint(upload))
    assert error.value.status_code == 413
    upload.read.assert_awaited_once_with(MAX_IMAGE_SIZE_BYTES + 1)


def test_endpoint_accepts_image_at_exact_size_limit(monkeypatch):
    from vintage_estimator.api import app as app_module
    from vintage_estimator.schemas import EstimationResult

    data = _fake_image_bytes()
    monkeypatch.setattr(app_module, "MAX_IMAGE_SIZE_BYTES", len(data))
    monkeypatch.setattr(app_module, "estimate", lambda image, top_k=5: EstimationResult(
        item="vase", style=None, price=None,
    ))
    response = client.post("/estimate", files={"file": ("photo.png", data, "image/png")})
    assert response.status_code == 200
