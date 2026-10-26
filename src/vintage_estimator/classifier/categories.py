

from __future__ import annotations

# каноническая категория -> промпты для текстового энкодера CLIP
CATEGORIES: dict[str, tuple[str, ...]] = {
    "Ваза": ("a photo of a decorative vase", "a vintage porcelain vase"),
    "Тарелка": ("a photo of a decorative collector plate", "a vintage porcelain plate"),
    "Блюдо": ("a vintage serving dish", "a decorative dish for serving food"),
    "Бокал": ("an antique goblet", "a vintage wine glass"),
    "Чашка": ("a vintage teacup", "an antique porcelain cup"),
    "Кружка": ("a vintage mug", "an old ceramic mug"),
    "Чайник": ("a vintage teapot", "an antique porcelain teapot"),
    "Кофейник": ("a vintage coffee pot", "an antique silver coffee pot"),
    "Кувшин": ("a vintage ceramic jug", "an antique pitcher"),
    "Графин": ("a vintage glass decanter", "an antique carafe"),
    "Бутылка": ("an antique bottle", "a vintage glass bottle"),
    "Сосуд": ("an antique metal vessel", "an old vessel"),
    "Стопка": ("a vintage shot glass", "an old small drinking glass"),
    "Молочник": ("a vintage milk jug", "an antique porcelain cream jug"),
    "Сахарница": ("a vintage sugar bowl", "an antique porcelain sugar bowl"),
    "Маслёнка": ("a vintage butter dish", "an antique porcelain butter dish"),
    "Солонка": ("a vintage salt cellar", "an antique salt shaker"),
    "Сервиз": ("a vintage tea set", "an antique porcelain dinnerware set"),
    "Поднос": ("a vintage serving tray", "an antique metal tray"),
    "Супница": ("a vintage soup tureen", "an antique porcelain tureen"),
    "Салатник": ("a vintage salad bowl", "an antique ceramic bowl"),
    "Фруктовница": ("a vintage fruit bowl", "an antique porcelain fruit vase"),
    "Статуэтка": ("a porcelain figurine", "an antique statuette"),
    "Кукла": ("a vintage doll", "an old collectible doll"),
    "Картина": ("an oil painting", "a framed painting"),
    "Гравюра": ("an antique engraving print", "an old lithograph artwork"),
    "Панно": ("a decorative wall plaque", "a vintage ceramic wall panel"),
    "Зеркало": ("an antique mirror", "a vintage wall mirror with a frame"),
    "Часы": ("an antique clock", "a vintage mantel clock"),
    "Лампа": ("a vintage table lamp", "an antique floor lamp"),
    "Подсвечник": ("a vintage candlestick", "an antique candelabra"),
    "Кашпо": ("a vintage flower pot holder", "an antique jardiniere planter"),
    "Полка": ("a vintage wall shelf", "an antique hanging shelf"),
    "Фоторамка": ("a vintage photo frame", "an antique picture frame"),
    "Украшение": ("an antique jewelry piece", "a vintage pendant"),
    "Ширма": ("an antique folding room screen", "a vintage room divider"),
    "Шкатулка": ("an antique box", "a vintage decorative casket box"),
    "Столовый прибор": ("a vintage silver cutlery piece", "an antique spoon"),
}

CATEGORY_NAMES: tuple[str, ...] = tuple(CATEGORIES)

# грязные метки датасета -> каноническая категория (ё/е и регистр не важны,
# см. canonical_category())
RAW_LABEL_TO_CATEGORY: dict[str, str] = {
    # посуда — тарелки и блюда
    "блюдо": "Блюдо",
    "селедочница": "Блюдо",
    "менажница": "Блюдо",
    "икорница": "Блюдо",
    "пашотница": "Блюдо",
    "медовница": "Блюдо",
    "терина": "Блюдо",
    "террина": "Блюдо",
    "бисквитница": "Блюдо",
    "тарелка": "Тарелка",
    # вазы
    "ваза": "Ваза",
    "вазы": "Ваза",
    "вазочка": "Ваза",
    "ваза-кубок": "Ваза",
    "ваза-кувшин": "Ваза",
    "ваза-солифлер": "Ваза",
    "ваза-фруктовница": "Ваза",
    # бокалы и мелкая посуда для питья
    "бокал": "Бокал",
    "бокал-флейта": "Бокал",
    "кубок": "Бокал",
    "стопка": "Стопка",
    "чашка": "Чашка",
    "кружка": "Кружка",
    # чайно-кофейные предметы
    "чайник": "Чайник",
    "кофейник": "Кофейник",
    "бульотка": "Супница",
    "супница": "Супница",
    "салатник": "Салатник",
    "фруктовница": "Фруктовница",
    "конфетница": "Сахарница",
    "тортница": "Сахарница",
    "чайница": "Сахарница",
    # кувшины и бутыли
    "кувшин": "Кувшин",
    "графин": "Графин",
    "бутылка": "Бутылка",
    "бутыль": "Бутылка",
    "фляга": "Бутылка",
    "штоф": "Бутылка",
    "ведро": "Сосуд",
    "лейка": "Сосуд",
    "сосуд": "Сосуд",
    "молочник": "Молочник",
    "соусник": "Молочник",
    "круэт": "Молочник",
    "масленка": "Маслёнка",
    "солонка": "Солонка",
    "сервиз": "Сервиз",
    "сервииз": "Сервиз",
    "набор": "Сервиз",
    "комплект": "Сервиз",
    "пара": "Сервиз",
    "трио": "Сервиз",
    "парные": "Сервиз",
    # поднос/полки/ящики
    "поднос": "Поднос",
    "лоток": "Поднос",
    "полка": "Полка",
    "вешалка-полка": "Полка",
    "коробка": "Шкатулка",
    # предметы интерьера
    "зеркало": "Зеркало",
    "часы": "Часы",
    "лампа": "Лампа",
    "торшер": "Лампа",
    "плафон": "Лампа",
    "подсвечник": "Подсвечник",
    "подсвечники": "Подсвечник",
    "канделябр": "Подсвечник",
    "вазочка-подсвечник": "Подсвечник",
    "кашпо": "Кашпо",
    "ширма": "Ширма",
    "фоторамка": "Фоторамка",
    # искусство
    "картина": "Картина",
    "гравюра": "Гравюра",
    "литография": "Гравюра",
    "автолитография": "Гравюра",
    "хромолитография": "Гравюра",
    "офорт": "Гравюра",
    "иллюстрация": "Гравюра",
    "панно": "Панно",
    "плакетка": "Панно",
    # скульптура
    "статуэтка": "Статуэтка",
    "скульптура": "Статуэтка",
    "бюст": "Статуэтка",
    "композиция": "Статуэтка",
    "сувенир": "Статуэтка",
    "бивень": "Статуэтка",
    "статуэтка-салфетница": "Статуэтка",
    "кукла-грелка": "Кукла",
    "кукла-хаката": "Кукла",
    "украшение": "Украшение",
    # столовые приборы
    "нож": "Столовый прибор",
    "лопатка": "Столовый прибор",
    "щипцы": "Столовый прибор",
    "прибор": "Столовый прибор",
    "орехокол": "Столовый прибор",
}

# обрывки заголовков, по которым категорию не восстановить
UNRECOGNIZED_LABELS = frozenset(
    {
        "авторский",
        "детский",
        "держатель",
        "емкость",
        "высокая",
        "декоративная",
        "декоративное",
        "дмитрий",
        "кофейная",
        "круглый",
        "миниатюрный",
        "малыш",
        "настенная",
        "настенное",
        "настольное",
        "основание",
        "подушка-подголовник",
        "фарфоровая",
        "скульптурная",
        "сонетка",
        "чайная",
        "чайно-кофейная",
        "чайное",
        "чайный",
        "сервировочное",
        "сервировочный",
    }
)


def canonical_category(raw: str | None) -> str | None:
    """Грязная метка из CSV -> каноническая категория.

    None — метку не распознали (мусор или пустая строка). Регистр и ё/е
    не различаются: «Селёдочница» == «Селедочница».
    """
    label = _norm(raw)
    if not label or label in UNRECOGNIZED_LABELS:
        return None
    if label in RAW_LABEL_TO_CATEGORY:
        return RAW_LABEL_TO_CATEGORY[label]
    return _category_from_suffix(label)


def _norm(raw: str | None) -> str:
    return (raw or "").strip().lower().replace("ё", "е")


def _category_from_suffix(label: str) -> str | None:
    """Составные метки вида «Ваза-кувшин», «Бокал-флейта»: ищем по первой части."""
    for part in label.split("-"):
        if part in RAW_LABEL_TO_CATEGORY:
            return RAW_LABEL_TO_CATEGORY[part]
    for part in label.split():
        if part in RAW_LABEL_TO_CATEGORY:
            return RAW_LABEL_TO_CATEGORY[part]
    return None
