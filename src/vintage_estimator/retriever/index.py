"""Переиспользуемый FAISS-индекс каталога с косинусным сходством."""

import faiss
import numpy as np

from vintage_estimator.schemas import Embedding, RetrievedItem

_index: faiss.Index | None = None
_items: list[RetrievedItem] = []


def load_catalog(embeddings: np.ndarray, items: list[RetrievedItem]) -> None:
    """Строит индекс поверх эмбедингов каталога (одна строка на предмет из items).

    Векторы нормализуются, поэтому inner product в FAISS даёт косинусное
    сходство — его же возвращает search() как RetrievedItem.score.
    """
    if embeddings.ndim != 2 or embeddings.shape[1] == 0:
        raise ValueError("ожидается матрица эмбедингов NxD, D > 0")
    if not np.isfinite(embeddings).all():
        raise ValueError("эмбединги должны быть конечными")
    if embeddings.shape[0] != len(items):
        raise ValueError("число эмбедингов не совпадает с числом предметов")

    normalized = np.array(embeddings, dtype=np.float32, order="C", copy=True)
    faiss.normalize_L2(normalized)

    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)

    global _index, _items
    _index = index
    _items = list(items)


def search(query: Embedding, top_k: int) -> list[RetrievedItem]:
    """top_k предметов каталога, ближайших к query по косинусному сходству.

    Пустой каталог — это не ошибка, а "похожих предметов пока нет":
    retriever.retrieve() должен уметь работать до того, как каталог загружен.
    """
    if top_k < 1:
        raise ValueError("top_k должен быть положительным")
    if _index is None or _index.ntotal == 0:
        return []

    if query.ndim != 1 or query.shape[0] != _index.d:
        raise ValueError(f"ожидается вектор размерности {_index.d}")
    if not np.isfinite(query).all():
        raise ValueError("запрос должен содержать конечные значения")
    q = np.array(query, dtype=np.float32, order="C", copy=True).reshape(1, -1)
    faiss.normalize_L2(q)

    k = min(top_k, _index.ntotal)
    scores, indices = _index.search(q, k)

    results: list[RetrievedItem] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = _items[idx]
        results.append(
            RetrievedItem(
                item_id=item.item_id,
                score=float(score),
                price=item.price,
                title=item.title,
                image_url=item.image_url,
            )
        )
    return results
