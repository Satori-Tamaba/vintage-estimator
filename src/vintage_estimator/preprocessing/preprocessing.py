import numpy as np
from PIL import Image as PILImage

from vintage_estimator.schemas import Image, ProcessedImage

TARGET_SIZE = 224

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
    # Прозрачность при этом теряется (фон становится чёрным) — прозрачные
    # фоны на фото антиквариата случаются редко, считать это фичей не стали.
    if pil.mode != "RGB":
        pil = pil.convert("RGB")

    resized = pil.resize((TARGET_SIZE, TARGET_SIZE), PILImage.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.uint8)


def _to_pil(image: Image) -> PILImage.Image:
    """Оформляет ndarray как PIL-картинку; несовместимые dtype приводит к uint8."""
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

    try:
        return PILImage.fromarray(arr)
    except TypeError:
        # fromarray берёт не все dtype (например, float64). Считаем, что
        # значения лежат в 0..255, и просто приводим тип — потеря точности
        uint8 = arr.astype(np.uint8) if arr.dtype.kind != "f" else np.clip(arr, 0, 255).astype(np.uint8)
        return PILImage.fromarray(uint8)
