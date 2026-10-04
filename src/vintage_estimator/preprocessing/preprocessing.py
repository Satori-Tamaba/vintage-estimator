from vintage_estimator.schemas import Image, ProcessedImage


def preprocess(image: Image) -> ProcessedImage:
    """Приводит сырую картинку пользователя к виду, который ждёт encoder.encode()."""
    raise NotImplementedError
