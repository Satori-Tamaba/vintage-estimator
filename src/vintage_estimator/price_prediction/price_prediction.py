import pandas as pd

from vintage_estimator.schemas import ClassificationResult, PricePrediction, RetrievedItem

# категория -> типичная цена по датасету, заполняется load_category_prices().
# Модуль-level состояние по той же причине, что и в retriever/index.py:
# контракт predict_price(classification, similar_items) не предполагает
# параметра с таблицей цен, а она общая для всех запросов.
_category_prices: dict[str, float] = {}


def load_category_prices(csv_path: str) -> None:
    """Строит таблицу "категория -> медианная цена" из CSV data_pipeline.

    Используется как fallback в predict_price(), когда у похожих предметов
    из retriever нет цены. Ожидает колонки category и price (см.
    data_pipeline/parse.py). Для готового каталога используем canon_category:
    эти же названия возвращает classifier, в отличие от сырых меток category.
    """
    df = pd.read_csv(csv_path)
    category_column = "canon_category" if "canon_category" in df.columns else "category"
    df = df.dropna(subset=[category_column, "price"])

    global _category_prices
    _category_prices = df.groupby(category_column)["price"].median().to_dict()


def predict_price(
    classification: ClassificationResult,
    similar_items: list[RetrievedItem],
) -> PricePrediction:
    """Оценивает цену предмета по похожим предметам из retriever.

    Если у similar_items нет цены (пустой список или ни у кого не указана),
    берём типичную цену по категории из _category_prices — это грубее, чем
    оценка по похожим вещам. Если нет и цены категории, возвращаем None.
    """
    priced_items = [item for item in similar_items if item.price is not None]
    if not priced_items:
        price = _category_prices.get(classification.category)
        return PricePrediction(price=price)

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
