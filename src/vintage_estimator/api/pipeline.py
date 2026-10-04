from vintage_estimator.classifier import classify
from vintage_estimator.encoder import encode
from vintage_estimator.preprocessing import preprocess
from vintage_estimator.price_prediction import predict_price
from vintage_estimator.retriever import retrieve
from vintage_estimator.schemas import EstimationResult, Image


def estimate(image: Image, top_k: int = 5) -> EstimationResult:
    """Прогоняет картинку через весь пайплайн и собирает финальный результат."""
    processed = preprocess(image)
    embedding = encode(processed)

    classification = classify(embedding)
    similar_items = retrieve(embedding, top_k=top_k)

    prediction = predict_price(classification, similar_items)

    return EstimationResult(
        item=classification.category,
        style=classification.style,
        price=prediction.price,
        similar_items=similar_items,
    )
