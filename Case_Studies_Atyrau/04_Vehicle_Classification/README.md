# Кейс №15 — классификация транспорта на COCO

Статус **PARTIAL**: реальные EDA, обучение и независимый от train тест выполнены; количественная оценка dashcam/Атырау отсутствует.

Данные: 669 объектов, 337 фотографий COCO val2017; train / validation / test = 403 / 133 / 133. HOG, RGB-гистограммы и отношение сторон; majority, LogisticRegression, RandomForest. Выбранная по validation логистическая модель: macro-F1 **0.4883**, accuracy **0.5489**; baseline macro-F1 **0.2362**. Accuracy baseline такая же. Классифицируется уже выделенный объект, а не весь дорожный кадр.

Откройте Report.pdf/Report.docx, Presentation.pdf/Presentation.pptx, Notebook.html, demo.mp4. Главные доказательства: dataset.csv, results/test_predictions.csv, results/metrics.json, results/classification_report.csv, results/training_history.csv, models/metadata.json. Переработанный текст слайдов — slides_v2.json; он генерируется из results скриптом scripts/build_slides_v2.py. slides.json оставлен как запасной формат. DOCX/PPTX/PDF создаются общим сборщиком пакета.

Из папки кейса, в окружении с requirements.txt:

```text
python scripts/prepare_data.py
python scripts/make_notebook.py
python scripts/make_artifacts.py
python scripts/build_slides_v2.py
python pipeline.py --verify-only
```

Повторная подготовка заново создаёт индекс без split; notebook детерминированно строит разбиение и выполняет реальное обучение. Для проверки существующих результатов подготовку не запускать. Последняя команда не переобучает модели: она извлекает признаки из изображений и проверяет сохранённые прогнозы. Для demo нужен ffmpeg на PATH. Интернет нужен при отсутствии raw-источников. Закреплённые хеши raw/source_integrity.json не обновляются автоматически; несоответствие или недоступность одного выбранного JPEG останавливает эксперимент. Это исключает скрытое изменение выборки при частичной загрузке.

Источники и лицензии — SOURCES.csv, References.md, results/image_attribution.csv. Фотографические производные сохраняют индивидуальные исходные CC-лицензии; часть допускает только некоммерческое использование. COCO не предоставил имён авторов в JSON; сохранены доступные прямые Flickr источники, отсутствующие реквизиты явно обозначены. Не считать все изображения CC BY4.0. Данные/проект не опубликованы.

demo.mp4:30 секунд реального CC0 dashcam Fernost, география неизвестна; боковые кропы выбраны вручную и имеют свои таймстампы. Это не тест accuracy и не обнаружение/трекинг. Нужна разрешённая размеченная локальная выборка и полный detector-пайплайн для закрытия первоначальной темы.
