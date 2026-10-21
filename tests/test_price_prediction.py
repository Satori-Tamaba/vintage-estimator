import pytest

from vintage_estimator.price_prediction import load_category_prices, predict_price
from vintage_estimator.price_prediction import price_prediction
from vintage_estimator.schemas import ClassificationResult, RetrievedItem

CLASSIFICATION = ClassificationResult(category="vase", style="art-deco", confidence=0.9)


@pytest.fixture(autouse=True)
def reset_category_prices():
    """Таблица категория->цена общая на модуль, чистим её после каждого теста."""
    yield
    price_prediction._category_prices = {}


def test_predict_price_returns_zero_when_no_similar_items():
    prediction = predict_price(CLASSIFICATION, [])

    assert prediction.price == 0.0
    assert prediction.price_range is None


def test_predict_price_ignores_items_without_price():
    items = [RetrievedItem(item_id="1", score=0.9, price=None)]

    prediction = predict_price(CLASSIFICATION, items)

    assert prediction.price == 0.0


def test_predict_price_weights_by_similarity_score():
    items = [
        RetrievedItem(item_id="close", score=0.9, price=1000.0),
        RetrievedItem(item_id="far", score=0.1, price=100.0),
    ]

    prediction = predict_price(CLASSIFICATION, items)

    # Более похожий предмет (score=0.9) должен тянуть цену к себе сильнее.
    assert prediction.price > 550.0
    assert prediction.price_range == (100.0, 1000.0)


def test_predict_price_falls_back_to_plain_average_when_scores_are_zero():
    items = [
        RetrievedItem(item_id="1", score=0.0, price=100.0),
        RetrievedItem(item_id="2", score=0.0, price=300.0),
    ]

    prediction = predict_price(CLASSIFICATION, items)

    assert prediction.price == 200.0


def test_predict_price_falls_back_to_category_price_when_no_similar_items(tmp_path):
    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text(
        "category,price\nvase,100\nvase,300\nclock,50\n", encoding="utf-8"
    )
    load_category_prices(str(csv_path))

    prediction = predict_price(CLASSIFICATION, [])

    assert prediction.price == 200.0  # медиана по vase
    assert prediction.price_range is None


def test_predict_price_falls_back_to_zero_for_unknown_category(tmp_path):
    csv_path = tmp_path / "catalog.csv"
    csv_path.write_text("category,price\nclock,50\n", encoding="utf-8")
    load_category_prices(str(csv_path))

    prediction = predict_price(CLASSIFICATION, [])  # CLASSIFICATION.category = "vase"

    assert prediction.price == 0.0
