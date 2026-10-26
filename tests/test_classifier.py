import numpy as np
import pytest

from vintage_estimator.classifier import canonical_category, classify
from vintage_estimator.classifier.categories import (
    CATEGORIES,
    CATEGORY_NAMES,
    RAW_LABEL_TO_CATEGORY,
    UNRECOGNIZED_LABELS,
)
from vintage_estimator.encoder import EMBEDDING_DIM

# --- чистка меток датасета: модель не нужна, гоняются всегда ---


def test_categories_table_is_consistent():
    """Целостность таблицы: у каждой категории есть непустые промпты,
    а маппинг не ссылается на неизвестные категории."""
    for name, prompts in CATEGORIES.items():
        assert name in CATEGORY_NAMES
        assert prompts and all(p.strip() for p in prompts)
    assert set(RAW_LABEL_TO_CATEGORY.values()) <= set(CATEGORY_NAMES)
    assert UNRECOGNIZED_LABELS


def test_canonical_category_maps_dirty_dataset_labels():
    assert canonical_category("Кружка") == "Кружка"
    assert canonical_category("Селёдочница") == canonical_category("Селедочница")
    assert canonical_category("сервииз") == "Сервиз"
    assert canonical_category("Ваза-солифлёр") == "Ваза"
    assert canonical_category("Кукла-хаката") == "Кукла"
    assert canonical_category("Маслёнка") == "Маслёнка"
    assert canonical_category("Малыш") is None  # обрывок заголовка
    assert canonical_category("") is None
    assert canonical_category(None) is None


# --- сам классификатор: нужны веса CLIP ---


def _sample_embedding(seed: int = 0) -> np.ndarray:
    """Детерминированный вектор единичной длины; сам CLIP не нужен —
    classify() перемножает матрицу text-эмбедингов на вектор."""
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return vector / np.linalg.norm(vector)


@pytest.mark.slow
def test_classify_returns_valid_category_and_confidence():
    result = classify(_sample_embedding())

    assert result.category in CATEGORY_NAMES
    assert 0.0 <= result.confidence <= 1.0
    assert result.style is None  # стили сознательно не определяем


@pytest.mark.slow
def test_classify_prefers_vector_warm_towards_its_prompts():
    """Вектор «близкий» к промптам категории даёт argmax этой категории:
    усредняем text-эмбединги промптов «Ваза» и подаём как image embedding."""
    from vintage_estimator.encoder import clip_model

    prompts = CATEGORIES["Ваза"]
    vase = clip_model.embed_texts(list(prompts)).mean(axis=0)
    vase /= np.linalg.norm(vase)

    result = classify(vase.astype(np.float32))
    assert result.category == "Ваза"
    assert result.confidence > 0.3


@pytest.mark.slow
def test_classify_rejects_wrong_dim():
    with pytest.raises(ValueError):
        classify(np.zeros(8, dtype=np.float32))
    with pytest.raises(ValueError):
        classify(np.zeros((2, EMBEDDING_DIM), dtype=np.float32))
