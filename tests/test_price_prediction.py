import pytest

from vintage_estimator.price_prediction import predict_price
from vintage_estimator.schemas import ClassificationResult


def test_predict_price_not_implemented_yet():
    """TODO: заменить на реальные тесты предсказания цены."""
    classification = ClassificationResult(category="vase", style="art-deco", confidence=0.9)
    with pytest.raises(NotImplementedError):
        predict_price(classification, [])
