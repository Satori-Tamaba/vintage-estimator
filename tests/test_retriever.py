import numpy as np
import pytest

from vintage_estimator.retriever import retrieve


def test_retrieve_not_implemented_yet():
    """TODO: заменить на реальные тесты ретривера."""
    with pytest.raises(NotImplementedError):
        retrieve(np.zeros(8, dtype=np.float32), top_k=3)
