import numpy as np
import pytest
from PIL import Image as PILImage

from vintage_estimator.preprocessing import TARGET_SIZE, preprocess


def test_preprocess_rgb_uint8_to_target_square():
    image = np.zeros((500, 300, 3), dtype=np.uint8)
    result = preprocess(image)

    assert result.shape == (TARGET_SIZE, TARGET_SIZE, 3)
    assert result.dtype == np.uint8


def test_preprocess_grayscale_2d_input():
    image = np.array(PILImage.new("L", (100, 50)))
    result = preprocess(image)

    assert result.shape == (TARGET_SIZE, TARGET_SIZE, 3)
    assert result.dtype == np.uint8


def test_preprocess_rgba_drops_alpha():
    image = np.array(PILImage.new("RGBA", (64, 64), color=(10, 20, 30, 255)))
    result = preprocess(image)

    assert result.shape == (TARGET_SIZE, TARGET_SIZE, 3)


def test_preprocess_float_input_scaled_from_0_255():
    image = np.zeros((30, 30, 3), dtype=np.float64)
    image[..., 0] = 128.0
    result = preprocess(image)

    assert result.dtype == np.uint8
    # красный канал примерно сохраняется при конвертации типа
    assert result.max() > 100


def test_preprocess_uint16_input():
    image = (np.ones((30, 30, 3)) * 40000).astype(np.uint16)
    result = preprocess(image)

    assert result.shape == (TARGET_SIZE, TARGET_SIZE, 3)
    assert result.dtype == np.uint8


def test_preprocess_is_deterministic():
    image = np.random.default_rng(0).integers(0, 256, (100, 200, 3), dtype=np.uint8)
    assert np.array_equal(preprocess(image), preprocess(image))


def test_preprocess_rejects_empty_and_wrong_shapes():
    with pytest.raises(ValueError):
        preprocess(np.zeros((0, 10, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        preprocess(np.zeros(5, dtype=np.uint8))
    with pytest.raises(ValueError):
        preprocess(np.zeros((10, 10, 7), dtype=np.uint8))
