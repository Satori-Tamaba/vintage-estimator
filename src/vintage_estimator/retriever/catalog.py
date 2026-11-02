"""Загрузка формата data_pipeline/build_embeddings.py без пересчёта векторов."""

from pathlib import Path

import numpy as np
import pandas as pd

from vintage_estimator.retriever.index import load_catalog
from vintage_estimator.schemas import RetrievedItem


def load_catalog_files(
    embeddings_path: str | Path,
    metadata_path: str | Path,
    expected_dim: int,
) -> None:
    """Строки CSV соответствуют строкам матрицы; lot_id сохраняется строкой."""
    with np.load(embeddings_path, allow_pickle=False) as archive:
        embeddings = archive["embeddings"]
    if embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
        raise ValueError(f"каталог должен содержать векторы размерности {expected_dim}")

    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig", dtype={"lot_id": str})
    required = {"lot_id", "price"}
    if not required.issubset(metadata.columns):
        raise ValueError("метаданные должны содержать lot_id и price")
    ids = metadata["lot_id"]
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("lot_id должны быть заполнены и уникальны")
    prices = pd.to_numeric(metadata["price"], errors="raise")
    known_prices = prices.dropna()
    if not np.isfinite(known_prices).all() or (known_prices < 0).any():
        raise ValueError("цены должны быть конечными и неотрицательными")

    items: list[RetrievedItem] = []
    for row, price in zip(metadata.to_dict("records"), prices):
        items.append(RetrievedItem(
            item_id=row["lot_id"], score=0.0,
            price=None if pd.isna(price) else float(price),
            title=None if pd.isna(row.get("title")) else str(row["title"]),
            image_url=None if pd.isna(row.get("image_url")) else str(row["image_url"]),
        ))
    load_catalog(embeddings, items)
