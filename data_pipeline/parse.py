"""Парсер раздела антиквариата на monetnik.ru.

Подраздел задаётся первым аргументом: короткое имя из SECTIONS, русское
название, путь или ссылка. Собирает лоты в data/monetnik_<имя>.csv и скачивает
первое фото каждого лота в data/raw_image/<id_лота>.jpg.

Запуск из корня проекта:
    uv add requests beautifulsoup4 pandas
    uv run python data_pipeline/parse_monetnik.py interier --limit 1000
    uv run python data_pipeline/parse_monetnik.py kuhnya --limit 1000
    uv run python data_pipeline/parse_monetnik.py "интерьер и декор"
    uv run python data_pipeline/parse_monetnik.py interier/vazy      # вложенный подраздел
    uv run python data_pipeline/parse_monetnik.py --list-sections

Скрипт можно прерывать и запускать снова: уже собранные лоты берутся из CSV.
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://www.monetnik.ru"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IMG_DIR = DATA_DIR / "raw_image"

# короткое имя -> (путь после /antikvariat/, имя для CSV-файла, русское название)
SECTIONS: dict[str, tuple[str, str, str]] = {
    "kuhnya": ("kuhnya", "kitchen", "Кухонная утварь"),
    "interier": ("interier", "interior", "Интерьер и декор"),
    "lichnoe": ("lichnoe", "personal", "Личные вещи"),
    "chasy": ("chasy", "clocks", "Часы"),
    "samovary": ("samovary", "samovars", "Самовары"),
    "tehnika": ("tehnika", "tech", "Техника"),
    "religia": ("religia", "religion", "Религиозная атрибутика, культовые предметы и реликвии"),
    "voennoe": ("voennoe", "military", "Военная и политическая атрибутика"),
    "pechatnoe": ("pechatnoe", "print", "Печатная продукция, документы и фотографии"),
    "igry": ("igry", "games", "Настольные, спортивные игры, модели и игрушки"),
    "odejda": ("odejda", "clothing", "Одежда и экипировка"),
}
# русские названия и их краткие формы
RU_ALIASES: dict[str, str] = {v[2].lower(): k for k, v in SECTIONS.items()}
RU_ALIASES.update(
    {
        "кухня": "kuhnya",
        "интерьер": "interier",
        "интерьер и декор": "interier",
        "религия": "religia",
        "военное": "voennoe",
        "печатная продукция": "pechatnoe",
        "игры": "igry",
        "игрушки": "igry",
    }
)

COLUMNS = [
    "image_url",
    "category",
    "price",
    "style",
    "era",
    "country",
    "title",
    "url",
    "condition",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9",
}

LATIN_TO_CYR = str.maketrans(
    {
        "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
        "O": "О", "P": "Р", "T": "Т", "X": "Х", "a": "а", "c": "с", "e": "е",
        "o": "о", "p": "р", "x": "х", "y": "у",
    }
)

log = logging.getLogger("monetnik")


# ------------------------------------------------------------ подраздел ----
def resolve_section(arg: str) -> tuple[str, str]:
    """Возвращает (путь подраздела, имя для файла).

    Принимает: 'interier', 'interier/vazy', 'Интерьер и декор',
    '/antikvariat/interier/', 'https://www.monetnik.ru/antikvariat/interier/'.
    """
    s = arg.strip().lower()
    s = RU_ALIASES.get(s, s)
    if s.startswith("http"):
        s = urlsplit(s).path
    s = s.strip("/")
    s = re.sub(r"^antikvariat(/|$)", "", s)
    s = re.sub(r"/?page\.\d+$", "", s)
    if not s:
        raise SystemExit("Укажите подраздел, например: interier (список: --list-sections)")
    if s in SECTIONS:
        return SECTIONS[s][0], SECTIONS[s][1]
    if not re.fullmatch(r"[a-z0-9\-]+(/[a-z0-9\-]+)*", s):
        raise SystemExit(f"Не понял подраздел «{arg}». Список: --list-sections")
    return s, s.replace("/", "_")


def category_url(path: str, page: int = 1) -> str:
    base = f"{BASE}/antikvariat/{path}/"
    return base if page == 1 else f"{base}page.{page}/"


# ----------------------------------------------------------------- сеть ----
def fetch(session: requests.Session, url: str, retries: int = 3) -> str | None:
    """Возвращает HTML или None, если страница не найдена / не загрузилась."""
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            log.warning("%s: попытка %d/%d не удалась: %s", url, attempt, retries, exc)
            time.sleep(2 * attempt)
    return None


def download_image(session: requests.Session, image_url: str, lot_id: str) -> bool:
    ext = Path(urlsplit(image_url).path).suffix.lower() or ".jpg"
    target = IMG_DIR / f"{lot_id}{ext}"
    if target.exists() and target.stat().st_size > 0:
        return True
    for attempt in range(1, 4):
        try:
            resp = session.get(image_url, timeout=30)
            resp.raise_for_status()
            if not resp.headers.get("Content-Type", "").startswith("image"):
                log.warning("%s: это не картинка", image_url)
                return False
            target.write_bytes(resp.content)
            return True
        except requests.RequestException as exc:
            log.warning("%s: попытка %d/3 не удалась: %s", image_url, attempt, exc)
            time.sleep(2 * attempt)
    return False


# -------------------------------------------------------------- разбор ----
def extract_lot_urls(html: str, section_path: str) -> list[str]:
    """Ссылки на лоты подраздела со страницы каталога (без дублей, по порядку).

    Лот — ссылка вида /antikvariat/<подраздел>/.../nazvanie-837604/ (id в конце).
    Ссылки на категории и теги (/t/...) под это не подпадают.
    """
    # теги вида .../interier/obihod/t/kashpo-1900/ отсекаются и в глубине пути
    def is_tag(path: str) -> bool:
        return "/t/" in path
    lot_re = re.compile(rf"^/antikvariat/{re.escape(section_path)}/(?!t/).+-\d{{5,}}/?$")
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(BASE, a["href"]).split("#")[0].split("?")[0]
        path = urlsplit(href).path
        if lot_re.match(path) and not is_tag(path) and href not in seen:
            seen.add(href)
            urls.append(href)
    return urls


def lot_id_from_url(url: str) -> str:
    m = re.search(r"-(\d+)/?$", urlsplit(url).path)
    return m.group(1) if m else re.sub(r"\W+", "_", url)


def first_word(title: str) -> str:
    """Первое слово названия: без префикса [..], знаков препинания, с поправкой
    латинских букв-двойников (на сайте встречается «Cупница» с латинской C)."""
    t = re.sub(r"^\s*(\[[^\]]*\]\s*)+", "", title)
    m = re.search(r"[^\W\d_]+(?:-[^\W\d_]+)*", t)
    word = m.group(0) if m else ""
    if re.search(r"[А-Яа-яЁё]", word) and re.search(r"[A-Za-z]", word):
        word = word.translate(LATIN_TO_CYR)
    return word


_KEY_RE = re.compile(r"^([^\W\d_][^:\n]{1,45}):\s*(.*)$")


def parse_props(soup: BeautifulSoup) -> dict[str, str]:
    """Характеристики лота: {ключ: первое значение}.

    Работает по тексту страницы, поэтому не зависит от конкретных тегов:
    строка «Ключ: значение» или «Ключ:» + значение(я) на следующих строках.
    Для многозначных полей (Стиль) берётся первое значение.
    """
    props: dict[str, str] = {}
    cur: str | None = None
    for line in soup.get_text("\n", strip=True).split("\n"):
        line = line.strip("  *•-")
        if not line:
            continue
        m = _KEY_RE.match(line)
        if m:
            key, val = m.group(1).strip(), m.group(2).strip()
            cur = key if key not in props else None  # первое вхождение побеждает
            if cur is not None:
                props[cur] = val
            continue
        if cur is not None and not props[cur]:
            props[cur] = line
    return props


def _abs(url: str) -> str:
    return "https:" + url if url.startswith("//") else urljoin(BASE, url)


def find_main_image(soup: BeautifulSoup) -> str | None:
    """Первое фото лота: og:image, запасной вариант — первая картинка галереи."""
    for attrs in ({"property": "og:image"}, {"name": "og:image"}):
        meta = soup.find("meta", attrs=attrs)
        if meta and meta.get("content") and "market-lot" in meta["content"]:
            return _abs(meta["content"])
    for a in soup.find_all("a", href=True):
        if "market-lot" in a["href"] and re.search(r"\.(jpe?g|png|webp)$", a["href"], re.I):
            return _abs(a["href"])
    return None


def find_price(soup: BeautifulSoup) -> int | None:
    """Текущая цена из <title> («... стоимостью 8889 руб.») или og:description."""
    title = soup.title.get_text() if soup.title else ""
    m = re.search(r"стоимостью\s+([\d\s ]+)\s*руб", title)
    if not m:
        meta = soup.find("meta", attrs={"property": "og:description"}) or soup.find(
            "meta", attrs={"name": "og:description"}
        )
        content = meta.get("content", "") if meta else ""
        m = re.search(r"за\s+([\d\s ]+)\s*руб", content)
    return int(re.sub(r"\D", "", m.group(1))) if m else None


def parse_lot(html: str, url: str) -> dict | None:
    """Разбор карточки лота. None — если у лота нет фото."""
    soup = BeautifulSoup(html, "html.parser")

    image_url = find_main_image(soup)
    if not image_url:
        return None

    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else ""
    if not title:
        meta = soup.find("meta", attrs={"property": "og:title"})
        title = meta["content"].strip() if meta and meta.get("content") else ""

    props = parse_props(soup)

    era = props.get("Год", "")
    if not era:  # запасной вариант: годы в конце названия, «..., 1910-1930 гг.»
        m = re.search(r"(\d{4}(?:\s*-\s*\d{4})?)\s*гг?\.?\s*$", title)
        era = m.group(1) if m else ""
    era = re.sub(r"\s*гг?\.?\s*$", "", era).replace(" ", "")

    return {
        "image_url": image_url,
        "category": first_word(title),
        "price": find_price(soup),
        "style": props.get("Стиль", ""),
        "era": era,
        "country": props.get("Страна", ""),
        "title": title,
        "url": url,
        "condition": props.get("Степень изношенности", ""),
    }


# ----------------------------------------------------------------- main ----
def save_csv(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("section", nargs="?", help="подраздел: interier, kuhnya, «интерьер и декор», interier/vazy, ссылка")
    ap.add_argument("--limit", type=int, default=1000, help="сколько лотов собрать (по умолчанию 1000)")
    ap.add_argument("--delay", type=float, default=0.5, help="пауза между запросами, сек")
    ap.add_argument("--max-pages", type=int, default=300, help="ограничение на страницы каталога")
    ap.add_argument("--out", type=Path, help="свой путь к CSV (по умолчанию data/monetnik_<имя>.csv)")
    ap.add_argument("--no-images", action="store_true", help="не скачивать картинки")
    ap.add_argument("--list-sections", action="store_true", help="показать известные подразделы и выйти")
    args = ap.parse_args()

    if args.list_sections:
        for key, (_, out_name, ru) in SECTIONS.items():
            print(f"{key:10} -> monetnik_{out_name}.csv   {ru}")
        print("\nЛюбой вложенный подраздел тоже подойдёт: interier/vazy, kuhnya/posuda ...")
        return
    if not args.section:
        ap.error("укажите подраздел, например: interier")

    section_path, out_name = resolve_section(args.section)
    csv_path = args.out or DATA_DIR / f"monetnik_{out_name}.csv"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Подраздел: /antikvariat/%s/  ->  %s", section_path, csv_path)

    rows: list[dict] = []
    if csv_path.exists():
        rows = pd.read_csv(csv_path).fillna("").to_dict("records")
        log.info("Продолжаем: в CSV уже %d лотов", len(rows))
    seen = {r["url"] for r in rows}

    session = requests.Session()
    session.headers.update(HEADERS)

    for page in range(1, args.max_pages + 1):
        if len(rows) >= args.limit:
            break
        page_url = category_url(section_path, page)
        html = fetch(session, page_url)
        time.sleep(args.delay)
        if html is None:
            log.info("Страница %d недоступна — конец каталога", page)
            break
        lot_urls = extract_lot_urls(html, section_path)
        if not lot_urls:
            log.info("На странице %d нет лотов — конец каталога", page)
            break
        log.info("Страница %d: %d лотов (собрано %d/%d)", page, len(lot_urls), len(rows), args.limit)

        for url in lot_urls:
            if len(rows) >= args.limit:
                break
            if url in seen:
                continue
            seen.add(url)
            lot_html = fetch(session, url)
            time.sleep(args.delay)
            if lot_html is None:
                continue
            item = parse_lot(lot_html, url)
            if item is None:
                log.info("Нет фото, пропускаем: %s", url)
                continue
            if not args.no_images and not download_image(session, item["image_url"], lot_id_from_url(url)):
                log.info("Фото не скачалось, пропускаем: %s", url)
                continue
            rows.append(item)
            if len(rows) % 25 == 0:
                save_csv(rows, csv_path)

    save_csv(rows, csv_path)
    log.info("Готово: %d лотов -> %s", len(rows), csv_path)


if __name__ == "__main__":
    main()