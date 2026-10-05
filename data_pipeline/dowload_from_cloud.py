import  os
import  zipfile
import  gdown

url = "https://drive.google.com/file/d/1HSM40tYboRt2zRNa49MCl0psTWvsABOF/view?usp=drive_link"
archive_name = "antiques_v1.zip"
destination_folder = "./data"
print("Загрузка архива")
gdown.download(url, archive_name)

import os
import zipfile
from pathlib import Path
import gdown

def download_and_extract_drive_zip(url: str, output_dir: Path | str) -> None:
    """
    Скачивает ZIP-архив с Google Диска и распаковывает его в указанную папку.
    """
    output_path = Path(output_dir)
    archive_path = output_path / "temp_downloaded_archive.zip"

    print(f"Загрузка архива...")
    os.makedirs(output_path, exist_ok=True)


    gdown.download(url, str(archive_path))

    if archive_path.exists():
        print(f"Распаковка zip в: {output_path}")
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)

        archive_path.unlink()
        print("Успешно завершено!")
    else:
        raise FileNotFoundError("Ошибка: Не удалось скачать файл с Google Диска.")



if __name__ == "__main__":

    GOOGLE_DRIVE_URL = url = "https://drive.google.com/file/d/1HSM40tYboRt2zRNa49MCl0psTWvsABOF/view?usp=drive_link"
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    TARGET_DATA_DIR = PROJECT_ROOT / "data"

    try:
        download_and_extract_drive_zip(url=GOOGLE_DRIVE_URL, output_dir=TARGET_DATA_DIR)
    except Exception as e:
        print(f"Произошла ошибка: {e}")

