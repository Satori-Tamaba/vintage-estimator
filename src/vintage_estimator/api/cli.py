import argparse
import json
import sys
from dataclasses import asdict
from zipfile import BadZipFile

import numpy as np
from PIL import Image as PILImage

from vintage_estimator.api.pipeline import estimate
from vintage_estimator.api.runtime import initialize_catalog


def main() -> None:
    parser = argparse.ArgumentParser(prog="vintage-estimator")
    parser.add_argument("image_path", help="путь к фото предмета")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--catalog-dir", help="каталог с embeddings.npz и embeddings_metadata.csv")
    parser.add_argument("--json", action="store_true", help="вывести результат в JSON")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k должен быть положительным")
    try:
        with PILImage.open(args.image_path) as uploaded:
            image = np.array(uploaded.convert("RGB"))
    except (OSError, ValueError, PILImage.DecompressionBombError) as exc:
        print(f"не удалось прочитать изображение: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.catalog_dir:
        try:
            initialize_catalog(args.catalog_dir)
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
            print(f"не удалось загрузить каталог: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

    result = estimate(image, top_k=args.top_k)
    print(json.dumps(asdict(result), ensure_ascii=False) if args.json else result)


if __name__ == "__main__":
    main()
