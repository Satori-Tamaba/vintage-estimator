"""Стык артефактов data_pipeline, запуска API, поиска и оценки цены."""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from vintage_estimator.api import pipeline
from vintage_estimator.api.app import app
from vintage_estimator.encoder import EMBEDDING_DIM
from vintage_estimator.price_prediction import price_prediction, predict_price
from vintage_estimator.retriever import index, retrieve
from vintage_estimator.retriever.catalog import load_catalog_files
from vintage_estimator.schemas import ClassificationResult


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    monkeypatch.setattr(index, "_index", None)
    monkeypatch.setattr(index, "_items", [])
    monkeypatch.setattr(price_prediction, "_category_prices", {})


def write_catalog(directory, matrix):
    np.savez_compressed(directory / "embeddings.npz", embeddings=matrix)
    (directory / "embeddings_metadata.csv").write_text(
        "lot_id,price,category,canon_category,title,image_url\n"
        "001,1200,Вазы,Ваза,Ваза,https://example.com/vase.png\n"
        "002,,Вазы,Ваза,Без цены,\n", encoding="utf-8-sig",
    )


def test_api_loads_catalog_at_startup(tmp_path, monkeypatch):
    matrix = np.eye(2, EMBEDDING_DIM, dtype=np.float32)
    write_catalog(tmp_path, matrix)
    monkeypatch.setenv("VINTAGE_CATALOG_DIR", str(tmp_path))
    monkeypatch.setattr(pipeline, "encode", lambda image: matrix[0].copy())
    monkeypatch.setattr(pipeline, "classify", lambda embedding: ClassificationResult(category="Ваза"))
    image = io.BytesIO()
    PILImage.new("RGB", (20, 30)).save(image, format="PNG")

    with TestClient(app) as client:
        response = client.post("/estimate", files={"file": ("photo.png", image.getvalue(), "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert result["price"] == 1200.0
    assert result["similar_items"][0]["item_id"] == "001"
    assert result["similar_items"][1]["price"] is None
    assert result["similar_items"][1]["image_url"] is None
    assert predict_price(ClassificationResult(category="Ваза"), []).price == 1200.0


@pytest.mark.parametrize("problem", ["dimension", "length", "duplicate", "negative", "infinite"])
def test_invalid_catalog_does_not_replace_index(tmp_path, problem):
    matrix = np.eye(2, EMBEDDING_DIM, dtype=np.float32)
    write_catalog(tmp_path, matrix)
    load_catalog_files(tmp_path / "embeddings.npz", tmp_path / "embeddings_metadata.csv", EMBEDDING_DIM)
    if problem == "dimension":
        np.savez(tmp_path / "embeddings.npz", embeddings=np.ones((2, 3)))
    elif problem == "length":
        np.savez(tmp_path / "embeddings.npz", embeddings=matrix[:1])
    else:
        path = tmp_path / "embeddings_metadata.csv"
        text = path.read_text(encoding="utf-8-sig")
        text = text.replace("002,", "001,") if problem == "duplicate" else text.replace("1200", "-1" if problem == "negative" else "inf")
        path.write_text(text, encoding="utf-8-sig")
    with pytest.raises(ValueError):
        load_catalog_files(tmp_path / "embeddings.npz", tmp_path / "embeddings_metadata.csv", EMBEDDING_DIM)
    assert retrieve(matrix[0], top_k=1)[0].item_id == "001"


def test_explicit_missing_catalog_fails_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("VINTAGE_CATALOG_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        with TestClient(app):
            pass


@pytest.mark.slow
def test_real_image_pipeline_with_catalog(tmp_path, monkeypatch):
    from vintage_estimator.encoder import encode
    from vintage_estimator.preprocessing import preprocess

    image = np.full((80, 100, 3), (120, 90, 60), dtype=np.uint8)
    image[20:60, 30:70] = (250, 240, 220)
    embedding = encode(preprocess(image))
    write_catalog(tmp_path, np.stack([embedding, -embedding]))
    monkeypatch.setenv("VINTAGE_CATALOG_DIR", str(tmp_path))
    photo = io.BytesIO()
    PILImage.fromarray(image).save(photo, format="PNG")
    with TestClient(app) as client:
        response = client.post("/estimate", files={"file": ("photo.png", photo.getvalue(), "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert result["item"]
    assert result["price"] == pytest.approx(1200.0)
    assert result["similar_items"][0]["item_id"] == "001"
    assert result["similar_items"][0]["score"] == pytest.approx(1.0, abs=1e-5)
