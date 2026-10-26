import numpy as np
import pytest

from PIL import Image as PILImage

from vintage_estimator.encoder import EMBEDDING_DIM, encode
from vintage_estimator.preprocessing import TARGET_SIZE, preprocess

pytestmark = pytest.mark.slow  # грузит веса CLIP (~350 МБ, кэш ~/.cache/huggingface)


def _sample_image() -> np.ndarray:
    """Не-однотонная картинка, чтобы embedding содержал реальный сигнал."""
    image = np.array(
        PILImage.new("RGB", (320, 240), color=(120, 90, 60))
    )
    image[50:150, 100:200] = (250, 240, 220)  # «предмет» на фоне
    return image


@pytest.fixture(scope="module")
def embedding():
    processed = preprocess(_sample_image())
    return encode(processed)


def test_encode_returns_normalized_vector_of_fixed_dim(embedding):
    assert embedding.shape == (EMBEDDING_DIM,)
    assert embedding.dtype == np.float32
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-3)


def test_encode_is_deterministic(embedding):
    processed = preprocess(_sample_image())
    assert np.allclose(encode(processed), embedding, atol=1e-5)


def test_encode_different_images_give_different_embeddings():
    a = preprocess(np.full((240, 240, 3), 0, dtype=np.uint8))
    b = preprocess(np.full((240, 240, 3), 255, dtype=np.uint8))
    emb_a = encode(a)
    emb_b = encode(b)
    assert np.linalg.norm(emb_a - emb_b) > 0.1


def test_encode_rejects_wrong_shape():
    with pytest.raises(ValueError):
        encode(np.zeros((224, 224), dtype=np.uint8))  # нет каналов
    with pytest.raises(ValueError):
        encode(np.zeros((10, 10, 7), dtype=np.uint8))
