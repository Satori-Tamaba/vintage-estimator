"""Общая инициализация каталога для HTTP и CLI."""

from pathlib import Path

from vintage_estimator.encoder import EMBEDDING_DIM
from vintage_estimator.price_prediction import load_category_prices
from vintage_estimator.retriever.catalog import load_catalog_files


def initialize_catalog(data_dir: str | Path) -> None:
    """Подключает готовые артефакты; отсутствие файлов — ошибка запуска."""
    directory = Path(data_dir)
    metadata_path = directory / "embeddings_metadata.csv"
    load_catalog_files(directory / "embeddings.npz", metadata_path, EMBEDDING_DIM)
    load_category_prices(str(metadata_path))
