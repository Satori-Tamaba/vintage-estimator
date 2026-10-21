"""Демо retriever + price_prediction без ожидания encoder/classifier от Дениса.

Каталог собран вручную (вместо реальных эмбедингов), чтобы показать, что
эти два модуля реально работают вместе: retriever находит похожие
предметы, price_prediction считает по ним цену, а если похожих предметов
нет - срабатывает fallback по категории из CSV (датасеты не коммитим,
см. .gitignore, поэтому CSV для примера создаётся тут же во временный файл).

Запуск из корня проекта: python examples/demo_retriever_price.py
"""

import tempfile
from pathlib import Path

import numpy as np

from vintage_estimator.price_prediction import load_category_prices, predict_price
from vintage_estimator.retriever import load_catalog, retrieve
from vintage_estimator.schemas import ClassificationResult, RetrievedItem

catalog_items = [
    RetrievedItem(item_id="1", score=0.0, price=1500.0, title="Самовар латунный"),
    RetrievedItem(item_id="2", score=0.0, price=3200.0, title="Самовар медный"),
    RetrievedItem(item_id="3", score=0.0, price=800.0, title="Часы настенные"),
]
catalog_embeddings = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
load_catalog(catalog_embeddings, catalog_items)

query = np.array([1.0, 0.05, 0.0], dtype=np.float32)  # "похож на самовар"
classification = ClassificationResult(category="samovar", style="классика", confidence=0.8)

print("--- сценарий 1: похожие предметы есть ---")
similar = retrieve(query, top_k=2)
for item in similar:
    print(f"  {item.title}: score={item.score:.3f}, price={item.price}")

prediction = predict_price(classification, similar)
print(f"цена: {prediction.price:.0f} руб, диапазон {prediction.price_range}")

print("\n--- сценарий 2: похожих предметов нет, fallback по категории ---")
with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
    f.write("category,price\nsamovar,1500\nsamovar,3200\nsamovar,2100\nclock,800\n")
    catalog_csv = f.name

load_category_prices(catalog_csv)
Path(catalog_csv).unlink()

prediction = predict_price(classification, [])
print(f"цена (медиана по категории '{classification.category}'): {prediction.price:.0f} руб")
