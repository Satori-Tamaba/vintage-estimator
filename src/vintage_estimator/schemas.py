"""Контракты между модулями пайплайна (см. imgs/img.png и project_structure.md).

Это единственное место, где описана форма данных, которой обмениваются
модули. Если меняется сигнатура здесь — нужно согласовать с владельцем
модуля по другую сторону контракта.
"""

from dataclasses import dataclass, field

import numpy as np

Image = np.ndarray
"""Сырая картинка от пользователя: HxWxC, uint8."""

ProcessedImage = np.ndarray
"""Картинка после preprocessing(), готова для encoder.encode()."""

Embedding = np.ndarray
"""Векторное представление картинки, выход encoder.encode().

Один и тот же эмбединг идёт и в classifier, и в retriever —
оба должны ожидать одинаковую размерность.
"""


@dataclass
class ClassificationResult:
    """Выход classifier.classify()."""

    category: str
    style: str | None = None
    confidence: float = 0.0


@dataclass
class RetrievedItem:
    """Один похожий предмет, найденный retriever.retrieve() в БД."""

    item_id: str
    score: float
    price: float | None = None
    title: str | None = None
    image_url: str | None = None


@dataclass
class PricePrediction:
    """Выход price_prediction.predict_price()."""

    price: float
    currency: str = "RUB"
    price_range: tuple[float, float] | None = None


@dataclass
class EstimationResult:
    """Финальный результат пайплайна, который API отдаёт пользователю."""

    item: str
    style: str | None
    price: float
    similar_items: list[RetrievedItem] = field(default_factory=list)
