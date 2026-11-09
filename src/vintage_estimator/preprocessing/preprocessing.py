from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps

from vintage_estimator.schemas import Image, ProcessedImage

TARGET_SIZE = 224


def load_image(source: str | Path | BinaryIO) -> Image:
    """Читает файл (путь или file-like) в HxWx3 uint8 RGB.

    Единая точка входа для CLI и data_pipeline: учитывает EXIF-поворот
    (фото с телефона) и приводит любой режим PIL (P, L, LA, RGBA, CMYK...)
    к RGB. Без этого np.asarray(PIL.Image) для палитровых картинок отдаёт
    2D-индексы палитры, и цвета теряются.
    """
    with PILImage.open(source) as opened:
        oriented = ImageOps.exif_transpose(opened)
        return np.asarray(oriented.convert("RGB"), dtype=np.uint8)


def preprocess(image: Image) -> ProcessedImage:
    """Приводит сырую картинку пользователя к виду, который ждёт encoder.encode().

    Вход — сырая картинка, HxW (или HxWxC), как её отдаёт np.array(PIL.Image).
    Выход — ProcessedImage: RGB uint8, TARGET_SIZE x TARGET_SIZE x 3.

    Масштабируем, а не обрезаем по центру: предметы на фото антиквариата часто
    вытянуты (вазы, канделябры), и сжатие сохраняет весь предмет, тогда как
    crop рискует отрезать верх/низ.
    """
    pil = _to_pil(image)

    # RGB-конверсия до ресайза: покрывает L/LA/P/RGBA за один шаг.
    # Альфа-канал при этом просто отбрасывается: прозрачные пиксели
    # сохраняют свой исходный RGB-цвет (это не обязательно чёрный).
    if pil.mode != "RGB":
        pil = pil.convert("RGB")

    resized = pil.resize((TARGET_SIZE, TARGET_SIZE), PILImage.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.uint8)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Приводит массив любого числового dtype к uint8 в диапазоне 0..255.

    - uint8 — без изменений;
    - uint16 — масштабируется (65535 -> 255), а не обрезается по модулю;
    - float — если все значения в 0..1, считаем это нормированной картинкой
      и умножаем на 255, иначе считаем, что значения уже в 0..255;
    - прочие целые и bool — зажимаются в 0..255.
    """
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * 255
    if arr.dtype == np.uint16:
        return (arr.astype(np.float32) / 257.0).round().astype(np.uint8)
    if arr.dtype.kind == "f":
        values = np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=255.0, neginf=0.0)
        if values.size and values.min() >= 0.0 and values.max() <= 1.0:
            values = values * 255.0
        return np.clip(values.round(), 0, 255).astype(np.uint8)
    if arr.dtype.kind in "iu":
        return np.clip(arr, 0, 255).astype(np.uint8)
    raise ValueError(f"неподдерживаемый dtype картинки: {arr.dtype}")


def _to_pil(image: Image) -> PILImage.Image:
    """Оформляет ndarray как PIL-картинку; любой dtype предварительно приводит к uint8."""
    arr = np.asarray(image)

    if arr.ndim == 3 and arr.shape[2] == 1:
        # одиночный канал равнозначен 2D-картинке: PIL fromarray его не берёт
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[2] not in (2, 3, 4):
        raise ValueError(f"неожиданная форма картинки: {arr.shape}, ожидаю HxW, HxWx2/3/4")
    if arr.ndim not in (2, 3):
        raise ValueError(f"картинка должна быть 2D (HxW) или 3D (HxWxC), пришло ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError("картинка пустая")

    return PILImage.fromarray(_to_uint8(arr))
