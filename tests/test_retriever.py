import numpy as np
import pytest

from vintage_estimator.retriever import load_catalog, retrieve
from vintage_estimator.retriever import index
from vintage_estimator.schemas import RetrievedItem


def test_retrieve_returns_empty_list_when_catalog_not_loaded():
    """До load_catalog() нет похожих предметов — это не ошибка пайплайна."""
    assert retrieve(np.zeros(4, dtype=np.float32), top_k=3) == []


def test_retrieve_finds_closest_items_by_cosine_similarity():
    catalog = [
        RetrievedItem(item_id="same", score=0.0, price=100.0, title="Такой же"),
        RetrievedItem(item_id="orthogonal", score=0.0, price=200.0, title="Другой"),
        RetrievedItem(item_id="opposite", score=0.0, price=300.0, title="Противоположный"),
    ]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    load_catalog(embeddings, catalog)

    results = retrieve(np.array([1.0, 0.0], dtype=np.float32), top_k=2)

    assert [item.item_id for item in results] == ["same", "orthogonal"]
    assert results[0].score > results[1].score
    assert results[0].price == 100.0


def test_retrieve_caps_top_k_to_catalog_size():
    catalog = [RetrievedItem(item_id="1", score=0.0, price=10.0)]
    load_catalog(np.array([[1.0, 0.0]], dtype=np.float32), catalog)

    results = retrieve(np.array([1.0, 0.0], dtype=np.float32), top_k=5)

    assert len(results) == 1


def test_load_catalog_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        load_catalog(np.zeros((2, 4), dtype=np.float32), [RetrievedItem(item_id="1", score=0.0)])


@pytest.fixture(autouse=True)
def reset_catalog(monkeypatch):
    monkeypatch.setattr(index, "_index", None)
    monkeypatch.setattr(index, "_items", [])


def test_normalization_does_not_change_callers_arrays():
    matrix = np.array([[3.0, 4.0]], dtype=np.float32)
    query = np.array([6.0, 8.0], dtype=np.float32)
    matrix.setflags(write=False)
    query.setflags(write=False)
    load_catalog(matrix, [RetrievedItem(item_id="1", score=0.0)])
    assert retrieve(query)[0].score == pytest.approx(1.0)
    np.testing.assert_array_equal(matrix, [[3.0, 4.0]])
    np.testing.assert_array_equal(query, [6.0, 8.0])


@pytest.mark.parametrize("matrix", [np.zeros(2), np.zeros((2, 0)), np.array([[np.nan]])])
def test_load_catalog_rejects_invalid_matrix(matrix):
    with pytest.raises(ValueError):
        load_catalog(matrix, [])


@pytest.mark.parametrize("top_k", [0, -1])
def test_retrieve_rejects_non_positive_top_k(top_k):
    with pytest.raises(ValueError):
        retrieve(np.ones(2), top_k=top_k)


def test_failed_reload_preserves_previous_catalog():
    load_catalog(np.array([[1.0, 0.0]]), [RetrievedItem(item_id="old", score=0.0)])
    with pytest.raises(ValueError):
        load_catalog(np.array([[np.nan, 0.0]]), [])
    assert retrieve(np.array([1.0, 0.0]))[0].item_id == "old"


@pytest.mark.parametrize("query", [np.ones(3), np.ones((1, 2)), np.array([np.inf, 0.0])])
def test_retrieve_rejects_invalid_query(query):
    load_catalog(np.array([[1.0, 0.0]]), [RetrievedItem(item_id="1", score=0.0)])
    with pytest.raises(ValueError):
        retrieve(query)
