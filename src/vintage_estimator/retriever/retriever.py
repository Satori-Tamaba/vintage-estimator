from vintage_estimator.schemas import Embedding, RetrievedItem


def retrieve(embedding: Embedding, top_k: int = 5) -> list[RetrievedItem]:
    """Ищет top_k ближайших по эмбедингу предметов в БД.

    Результат идёт в price_prediction.predict_price() как опорные
    примеры для оценки цены и в финальный результат как "похожие предметы".
    """
    raise NotImplementedError
