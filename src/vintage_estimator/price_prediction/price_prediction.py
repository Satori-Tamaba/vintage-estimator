from vintage_estimator.schemas import ClassificationResult, PricePrediction, RetrievedItem


def predict_price(
    classification: ClassificationResult,
    similar_items: list[RetrievedItem],
) -> PricePrediction:
    """Оценивает цену предмета по похожим предметам из retriever.

    classification пока не используется: эмбединг, по которому retriever
    нашёл similar_items, уже неявно учитывает категорию и стиль предмета
    (похожие по эмбедингу вещи обычно похожи и по категории). Параметр
    зафиксирован контрактом в schemas.py и понадобится, когда появится
    датасет с ценами по категориям — для fallback, если у similar_items
    нет цены (см. ветку ниже).
    """
    priced_items = [item for item in similar_items if item.price is not None]
    if not priced_items:
        return PricePrediction(price=0.0)

    weights = [max(item.score, 0.0) for item in priced_items]
    total_weight = sum(weights)

    if total_weight == 0.0:
        # Все score нулевые или отрицательные — веса не дают сигнала,
        # используем простое среднее по ценам, а не деление на ноль.
        price = sum(item.price for item in priced_items) / len(priced_items)
    else:
        price = sum(
            weight * item.price for weight, item in zip(weights, priced_items)
        ) / total_weight

    prices = [item.price for item in priced_items]
    return PricePrediction(price=price, price_range=(min(prices), max(prices)))
