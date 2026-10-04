from vintage_estimator.schemas import Embedding, ProcessedImage


def encode(image: ProcessedImage) -> Embedding:
    """Строит эмбединг обработанной картинки.

    Результат потребляют classifier.classify() и retriever.retrieve() —
    у них должна совпадать ожидаемая размерность вектора.
    """
    raise NotImplementedError
