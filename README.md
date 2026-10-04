# vintage-estimator

Проект: оценка антиквариата по фото

## Команда проекта:
@kustarevv - https://github.com/Kustarevvv  
@SatoriTanabi - https://github.com/Satori-Tamaba

## Схема проекта:
![Модули](imgs/img.png)
Зона ответственности:
Препроцессинг энкодер, классификатор - Зеленин Денис  
Ретривер, предсказание цены api - Кустарев Александр

## Структура проекта

```
src/vintage_estimator/
├── schemas.py          # общие типы данных между модулями
├── preprocessing/
├── encoder/
├── classifier/
├── retriever/
├── price_prediction/
└── api/                 # собирает пайплайн: estimate(image) -> результат
tests/
```

Подробнее про контракты между модулями — в `project_structure.md`.

