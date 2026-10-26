import numpy as np
import torch

from vintage_estimator.encoder import clip_model
from vintage_estimator.schemas import Embedding, ProcessedImage

EMBEDDING_DIM = clip_model.EMBEDDING_DIM
"""Размерность эмбединга; тот же контракт, что описан в clip_model.py."""


def encode(image: ProcessedImage) -> Embedding:
    """Строит эмбединг обработанной картинки.

    Результат потребляют classifier.classify() и retriever.retrieve() —
    у них должна совпадать ожидаемая размерность вектора.

    Вход: ProcessedImage из preprocessing.preprocess() — RGB uint8,
    TARGET_SIZE x TARGET_SIZE x 3. Выход: np.ndarray float32,
    shape (EMBEDDING_DIM,), L2-нормализованный.

    Нормализуем вектор здесь, а не на стороне потребителя: и retriever
    (косинусное сходство через FAISS), и classifier (zero-shot) сравнивают
    эмбединги через скалярное произведение — это корректно, только если
    все векторы лежат на единичной сфере.
    """
    batch = _validate_batch([image])
    return _encode_batch(batch)[0]


def encode_batch(images: list[ProcessedImage]) -> list[Embedding]:
    """То же, что encode(), но пачкой один forward: конвейер заметно быстрее,
    когда каталог эмбедингов для retriever считается на сотнях фото."""
    batch = _validate_batch(images)
    return _encode_batch(batch)


def _validate_batch(images: list[ProcessedImage]) -> np.ndarray:
    if not images:
        raise ValueError("пустой список картинок")
    arrays = []
    for i, image in enumerate(images):
        shape = getattr(image, "shape", None)
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(
                f"encode() ждёт ProcessedImage HxWx3 из preprocess(); "
                f"картинка {i}: shape={shape}"
            )
        arrays.append(arr)
    first = arrays[0].shape
    if any(arr.shape != first for arr in arrays):
        raise ValueError(
            f"все картинки должны быть одного размера HxWx3 (выход preprocess()), "
            f"первая: {first}"
        )
    return np.ascontiguousarray(np.stack(arrays))


def _encode_batch(batch: np.ndarray) -> list[Embedding]:
    model, _, device = clip_model.get_model()

    # HWC uint8 (0..255) -> CHW float32 (0..1) -> CLIP mean/std.
    # Ресайз уже сделан в preprocess(), поэтому transform из open_clip
    # не используем — здесь остаётся только нормализация пикселей.
    tensor = torch.from_numpy(batch.astype(np.float32))
    tensor = tensor.permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor(clip_model.CLIP_MEAN, dtype=torch.float32, device=device)
    std = torch.tensor(clip_model.CLIP_STD, dtype=torch.float32, device=device)
    tensor = (tensor - mean[None, :, None, None]) / std[None, :, None, None]

    with torch.no_grad():
        features = model.encode_image(tensor.to(device))

    features = features / features.norm(dim=-1, keepdim=True)
    features = features.float().cpu().numpy()
    return [row for row in features]
