import numpy as np
import pytest

from vintage_estimator.preprocessing import preprocess


def test_preprocess_not_implemented_yet():
    """TODO: заменить на реальные тесты препроцессинга."""
    with pytest.raises(NotImplementedError):
        preprocess(np.zeros((4, 4, 3), dtype=np.uint8))
