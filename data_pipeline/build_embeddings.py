"""Каталог эмбедингов для retriever: фото из data/raw_image -> векторы + метаданные.

Зачем: retriever ищет похожие предметы сравнением векторов, но векторы нужно
заранее посчитать по всему датасету (это работа encoder'а — здесь Денис).
Результат кладётся на диск и подключается через retriever.load_catalog():

    import numpy as np, pandas as pd
    from vintage_estimator.retriever import load_catalog
    from vintage_estimator.schemas import RetrievedItem

    data = np.load("data/embeddings.npz")
    meta = pd.read_csv("data/embeddings_metadata.csv", encoding="utf-8-sig")
    load_catalog(data["embeddings"], [
        RetrievedItem(item_id=row.lot_id, price=row.price, title=row.title,
                      image_url=row.image_url)
        for row in meta.itertuples()
    ])

Запуск:
    uv run python data_pipeline/build_embeddings.py
    uv run python data_pipeline/build_embeddings.py --csv data/monetnik_kitchen.csv

Скрипт можно прерывать и запускать снова: готовый .npz дозаписывается
по lot_id, повторная генерация считает только новые лоты.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vintage_estimator.classifier import canonical_category  # noqa: E402
from vintage_estimator.encoder import encode_batch, EMBEDDING_DIM  # noqa: E402
from vintage_estimator.preprocessing import preprocess  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IMG_DIR = DATA_DIR / "raw_image"

EMBEDDINGS_NPZ = DATA_DIR / "embeddings.npz"
METADATA_CSV = DATA_DIR / "embeddings_metadata.csv"

LOT_ID_RE = re.compile(r"-(\d+)/?$")

log = logging.getLogger("build_embeddings")


def lot_id_from_url(url: str) -> str:
    """id лота из monetnik-ссылки: .../vaza-nazvanie-837604/ -> 837604.

    Это же правило использовал parse.py, когда называл файлы в raw_image.
    """
    match = LOT_ID_RE.search((url or "").strip())
    return match.group(1) if match else ""


def load_rows(csv_paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p, encoding="utf-8-sig") for p in csv_paths]
    rows = pd.concat(frames, ignore_index=True)
    rows["lot_id"] = rows["url"].map(lot_id_from_url)
    # дубли при пересечении разделов: фото одно и то же, эмбединг не нужен дважды
    rows = rows.drop_duplicates(subset="lot_id", keep="first")
    return rows


def image_path(lot_id: str) -> Path | None:
    for extension in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = IMG_DIR / f"{lot_id}{extension}"
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def load_image(path: Path) -> np.ndarray | None:
    try:
        return np.asarray(PILImage.open(path))
    except Exception as exc:
        log.warning("%s: не читается (%s)", path.name, exc)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", nargs="*", type=Path,
        help="CSV-файлы Монетника (по умолчанию: все data/monetnik_*.csv)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--out-npz", type=Path, default=EMBEDDINGS_NPZ,
        help="куда писать матрицу эмбедингов",
    )
    parser.add_argument(
        "--out-csv", type=Path, default=METADATA_CSV,
        help="куда писать метаданные (одна строка на строку матрицы)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    csv_paths = args.csv or sorted(DATA_DIR.glob("monetnik_*.csv"))
    if not csv_paths:
        raise SystemExit(f"нет CSV-файлов в {DATA_DIR} — сначала запустите data_pipeline/parse.py")

    rows = load_rows(csv_paths)
    log.info("Лотов в CSV: %d (%s)", len(rows), [p.name for p in csv_paths])

    prev_matrix, prev_meta = load_done(args.out_npz, args.out_csv)
    already_done = set(prev_meta["lot_id"].astype(str)) if not prev_meta.empty else set()
    todo = rows[~rows["lot_id"].isin(already_done)].reset_index(drop=True)
    total = len(already_done) + len(todo)
    log.info("Уже посчитано: %d, осталось: %d", len(already_done), len(todo))
    if todo.empty:
        log.info("Каталог актуален")
        return

    # старый каталог продолжаем дописывать, а не переписываем заново
    embeddings = [row for row in prev_matrix] if prev_matrix is not None else []
    metadata = prev_meta.to_dict("records") if not prev_meta.empty else []
    for start in range(0, len(todo), args.batch_size):
        chunk = todo.iloc[start : start + args.batch_size]

        images = []
        for lot_id in chunk["lot_id"]:
            path = image_path(lot_id)
            raw = load_image(path) if path else None
            images.append(preprocess(raw) if raw is not None else None)

        valid_indices = [i for i, img in enumerate(images) if img is not None]
        if not valid_indices:
            log.warning("Батч %d..%d: ни одного фото — пропускаем", start, start + len(chunk))
            continue

        encoded = encode_batch([images[i] for i in valid_indices])

        for i, embedding in zip(valid_indices, encoded):
            row = chunk.iloc[i]
            lot_id = str(row["lot_id"])
            embeddings.append(embedding)
            metadata.append(
                {
                    "lot_id": lot_id,
                    "price": row.get("price"),
                    "category": row.get("category"),
                    "canon_category": canonical_category(row.get("category") or ""),
                    "style": row.get("style"),
                    "era": row.get("era"),
                    "country": row.get("country"),
                    "title": row.get("title"),
                    "image_url": row.get("image_url"),
                }
            )

        if (start // args.batch_size) % 10 == 0 or start + args.batch_size >= len(todo):
            log.info("Посчитано %d/%d", len(metadata), total)
            save(args.out_npz, args.out_csv, embeddings, metadata)

    save(args.out_npz, args.out_csv, embeddings, metadata)
    log.info(
        "Готово: %d векторов dim=%d -> %s + %s",
        len(metadata), EMBEDDING_DIM, args.out_npz, args.out_csv,
    )


def load_done(npz_path: Path, csv_path: Path) -> tuple[np.ndarray | None, pd.DataFrame]:
    """Каталог на диске: (матрица Nx512, метаданные Nx1) — если файлы согласованы."""
    if not npz_path.exists() or not csv_path.exists():
        return None, pd.DataFrame()
    try:
        with np.load(npz_path) as npz:
            matrix = npz["embeddings"]
    except Exception as exc:
        log.warning("Не удалось прочитать %s (%s) — считаем каталог с нуля", npz_path.name, exc)
        return None, pd.DataFrame()

    meta = pd.read_csv(csv_path, encoding="utf-8-sig").fillna("")
    if len(meta) != matrix.shape[0]:
        log.warning("%s (%d строк) не совпадает по длине с %s (%d векторов) — пересчёт с нуля",
                    csv_path.name, len(meta), npz_path.name, matrix.shape[0])
        return None, pd.DataFrame()
    if matrix.shape[1] != EMBEDDING_DIM:
        log.warning("Старая размерность %d, а сейчас %d — пересчёт с нуля",
                    matrix.shape[1], EMBEDDING_DIM)
        return None, pd.DataFrame()
    return matrix.astype(np.float32), meta


def save(npz_path: Path, csv_path: Path, embeddings: list[np.ndarray], metadata: list[dict]) -> None:
    if not metadata:
        return
    np.savez_compressed(npz_path, embeddings=np.stack(embeddings))
    pd.DataFrame(metadata).to_csv(csv_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
