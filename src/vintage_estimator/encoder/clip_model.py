"""Общая ленивая загрузка CLIP (ViT-B-32, weights "openai") для encoder и classifier.

Оба модуля работают с одной и той же сетью: encode_image() нужен encoder'у,
encode_text() — zero-shot классификатору. Держать две копии весов в памяти —
роскошь, поэтому загрузка общая: первый вызов get_model() скачивает/читает
веса (~350 МБ, кэш ~/.cache/huggingface), последующие — мгновенны.

Загрузка ленивая и потокобезопасная: импортировать модули должно быть дёшево
(это делают и API, и CLI), а модель создаётся один раз на процесс.
"""

import threading

import numpy as np
import open_clip
import torch

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
EMBEDDING_DIM = 512
"""Размерность выхода visual и text encoder'а ViT-B-32.

Контракт с retriever (retriever/index.py): FAISS-индекс строится от
embeddings.shape[1], а каталог эмбедингов считается заранее. Смена модели
(например, на ViT-B-16, там 768) означает пересчёт всего каталога.
"""

# Нормализация openai-CLIP (константы из репозитория open_clip; хардкод,
# чтобы не зависеть от того, как они экспортируются в конкретной версии).
CLIP_MEAN = (0.48145466, 0.45742791, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_lock = threading.Lock()
_model: "open_clip.model.CLIP | None" = None
_tokenizer = None
_device: str | None = None


def get_model() -> tuple["open_clip.model.CLIP", object, str]:
    """Возвращает (model, tokenizer, device). Модель в eval, без градиентов."""
    global _model, _tokenizer, _device
    if _model is None:
        with _lock:
            if _model is None:  # двойная проверка: другой поток мог успеть
                model, _preprocess, _ = open_clip.create_model_and_transforms(
                    MODEL_NAME, pretrained=PRETRAINED
                )
                model.eval()
                _model = model
                _tokenizer = open_clip.get_tokenizer(MODEL_NAME)
                _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _model, _tokenizer, _device


def embed_texts(texts: list[str]) -> np.ndarray:
    """Текстовые эмбединги (N x EMBEDDING_DIM), L2-нормализованные, float32.

    Нормализация — та же схема, что у encode() для картинок: оба сравниваются
    по косинусному сходству, и с нормализованными векторами это просто
    скалярное произведение.
    """
    model, tokenizer, device = get_model()
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.float().cpu().numpy()
