import numpy as np
import pytest

from vintage_estimator.classifier import classify


def test_classify_not_implemented_yet():
    """TODO: заменить на реальные тесты классификатора."""
    with pytest.raises(NotImplementedError):
        classify(np.zeros(8, dtype=np.float32))
