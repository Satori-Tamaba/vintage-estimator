import numpy as np
import pytest

from vintage_estimator.encoder import encode


def test_encode_not_implemented_yet():
    """TODO: заменить на реальные тесты энкодера."""
    with pytest.raises(NotImplementedError):
        encode(np.zeros((4, 4, 3), dtype=np.float32))
