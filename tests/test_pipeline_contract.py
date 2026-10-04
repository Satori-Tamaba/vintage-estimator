"""Доказывает, что контракты между модулями стыкуются, не дожидаясь реализации.

Каждая стадия пайплайна подменяется заглушкой, которая возвращает
данные нужной формы (см. schemas.py). Если кто-то поменяет сигнатуру
своей функции несовместимо с этим тестом — пайплайн сломается здесь,
а не в рантайме у пользователя.
"""

import numpy as np

from vintage_estimator.api import pipeline
from vintage_estimator.schemas import ClassificationResult, PricePrediction, RetrievedItem


def test_pipeline_wires_all_stages_together(monkeypatch):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    processed = np.zeros((2, 2, 3), dtype=np.float32)
    embedding = np.zeros(8, dtype=np.float32)
    classification = ClassificationResult(category="vase", style="art-deco", confidence=0.9)
    similar_items = [RetrievedItem(item_id="1", score=0.95, price=1000.0, title="Ваза")]
    prediction = PricePrediction(price=1200.0)

    monkeypatch.setattr(pipeline, "preprocess", lambda img: processed)
    monkeypatch.setattr(pipeline, "encode", lambda img: embedding)
    monkeypatch.setattr(pipeline, "classify", lambda emb: classification)
    monkeypatch.setattr(pipeline, "retrieve", lambda emb, top_k=5: similar_items)
    monkeypatch.setattr(pipeline, "predict_price", lambda cls, items: prediction)

    result = pipeline.estimate(image, top_k=3)

    assert result.item == "vase"
    assert result.style == "art-deco"
    assert result.price == 1200.0
    assert result.similar_items == similar_items
