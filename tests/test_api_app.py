"""Проверяет HTTP-слой отдельно от нереализованных стадий пайплайна.

Стадии подменяются заглушками (как в test_pipeline_contract.py), чтобы
тест проверял именно API: приём файла, вызов estimate(), сериализацию ответа.
"""

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from vintage_estimator.api import pipeline
from vintage_estimator.api.app import app
from vintage_estimator.schemas import ClassificationResult, PricePrediction, RetrievedItem

client = TestClient(app)


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
