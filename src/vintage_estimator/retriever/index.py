"""In-memory векторный индекс над каталогом предметов (FAISS, косинусное сходство).

Пока нет готового датасета с реальными эмбедингами (Денис ещё не отдал
encoder, data_pipeline/ только собирает фото и цены — без эмбедингов),
индекс строится через load_catalog() вручную, в т.ч. из тестов. Как
только появится CSV с эмбедингами — загрузка каталога подключится сюда
без изменений в retriever.retrieve().

Индекс и список items хранятся в module-level переменных, а не передаются
параметром — это осознанное нарушение правила "без глобального состояния"
из CONTRIBUTING.md: контракт retrieve(embedding, top_k) в schemas.py/pipeline.py
не предполагает параметра с каталогом, а строить FAISS-индекс заново на
каждый запрос слишком дорого. Если каталог понадобится переключать в
рантайме (например, разные категории в разных индексах) — это повод
завести класс Retriever с явным состоянием, а не продолжать расширять
глобальные переменные.
"""

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
    if embeddings.shape[0] != len(items):
        raise ValueError("число эмбедингов не совпадает с числом предметов")

    normalized = np.ascontiguousarray(embeddings, dtype=np.float32)
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
    if _index is None or _index.ntotal == 0:
        return []

    q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
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
