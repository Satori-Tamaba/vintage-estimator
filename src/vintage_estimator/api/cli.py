import argparse
import sys

import numpy as np
from PIL import Image as PILImage

from vintage_estimator.api.pipeline import estimate


def main() -> None:
    parser = argparse.ArgumentParser(prog="vintage-estimator")
    parser.add_argument("image_path", help="путь к фото предмета")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    try:
        image = np.array(PILImage.open(args.image_path))
    except FileNotFoundError:
        print(f"файл не найден: {args.image_path}", file=sys.stderr)
        raise SystemExit(1)

    result = estimate(image, top_k=args.top_k)
    print(result)


if __name__ == "__main__":
    main()
