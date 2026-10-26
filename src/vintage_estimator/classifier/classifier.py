"""Zero-shot классификатор категории на CLIP.

Обучение не нужно: image embedding (из encoder.encode()) сравнивается с
текстовыми эмбедингами промптов из classifier.categories.CATEGORIES.
Побеждает категория с максимальным косинусным сходством; confidence —
softmax по сходствам (температура — logit_scale модели, у openai-CLIP ~100).

Текстовые эмбединги считаются один раз при первом вызове classify() и
кэшируются: им нужно по одному forward на промпт, а промпты не меняются.
"""

import threading

import numpy as np

from vintage_estimator.classifier import categories
from vintage_estimator.encoder import clip_model
from vintage_estimator.schemas import ClassificationResult, Embedding

_lock = threading.Lock()
_text_names: tuple[str, ...] | None = None
_text_matrix: np.ndarray | None = None
_logit_scale: float | None = None


def classify(embedding: Embedding) -> ClassificationResult:
    """Определяет категорию предмета по эмбедингу.

    Результат идёт в price_prediction.predict_price() как контекст
    для оценки цены. style не определяем (широкие «стили» Монетника —
    «Античный», «Модерн»... — без явного списка промптов zero-shot не
    ловит); schema допускает style=None.
    """
    text_names, text_matrix, logit_scale = _text_embeddings()

    vector = np.asarray(embedding, dtype=np.float32)
    if vector.ndim != 1 or vector.shape[0] != text_matrix.shape[1]:
        raise ValueError(
            f"classify() ждёт эмбединг размерности {text_matrix.shape[1]} "
            f"из encoder.encode(), пришло shape={vector.shape}"
        )

    similarities = text_matrix @ vector
    probabilities = _softmax(similarities * logit_scale)

    best = int(np.argmax(probabilities))
    return ClassificationResult(
        category=text_names[best],
        style=None,
        confidence=float(probabilities[best]),
    )


def _text_embeddings() -> tuple[tuple[str, ...], np.ndarray, float]:
    """(имена категорий, матрица эмбедингов, logit_scale) — лениво, один раз.

    Матрица усредняет промпты внутри категории (по 2 промпта на категорию,
    см. categories.CATEGORIES) и L2-нормализует строки — тот же формат,
    в котором encode() отдаёт image embedding.
    """
    global _text_names, _text_matrix, _logit_scale
    if _text_matrix is None:
        with _lock:
            if _text_matrix is None:  # двойная проверка: другой поток мог успеть
                prompts: list[str] = []
                names: list[str] = []
                for name, category_prompts in categories.CATEGORIES.items():
                    names.append(name)
                    prompts.extend(category_prompts)

                model, _, _device = clip_model.get_model()
                prompt_embeddings = clip_model.embed_texts(prompts)

                averaged = []
                row = 0
                for name in names:
                    n = len(categories.CATEGORIES[name])
                    averaged.append(prompt_embeddings[row : row + n].mean(axis=0))
                    row += n
                matrix = np.stack(averaged)
                matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

                _text_names = tuple(names)
                _text_matrix = matrix.astype(np.float32)
                _logit_scale = float(model.logit_scale.exp().item())
    return _text_names, _text_matrix, _logit_scale


def _softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - x.max())
    return exp / exp.sum()
