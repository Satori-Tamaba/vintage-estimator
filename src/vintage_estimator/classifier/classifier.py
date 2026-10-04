from vintage_estimator.schemas import ClassificationResult, Embedding


def classify(embedding: Embedding) -> ClassificationResult:
    """Определяет категорию (и по возможности стиль) предмета по эмбедингу.

    Результат идёт в price_prediction.predict_price() как контекст
    для оценки цены.
    """
    raise NotImplementedError
