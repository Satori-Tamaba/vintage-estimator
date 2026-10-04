from vintage_estimator.schemas import ClassificationResult, PricePrediction, RetrievedItem


def predict_price(
    classification: ClassificationResult,
    similar_items: list[RetrievedItem],
) -> PricePrediction:
    """Оценивает цену предмета по его категории и похожим предметам из БД."""
    raise NotImplementedError
