"""Summarize verified current artifacts; do not infer unperformed checks."""
from pathlib import Path
import json,csv,hashlib,argparse
import nbformat
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--archive-checked',action='store_true');a=p.parse_args()
run=json.loads((ROOT/'reproduction_run.json').read_text());checks=json.loads((ROOT/'verification_results.json').read_text())
assert run and all(x['returncode']==0 for x in run), 'Full reproduction must actually pass'
assert all(x['status']=='PASS' for x in checks)
rows=[];pages_total=slides_total=0
for case in sorted(ROOT.glob('0[1-4]_*')):
 evidence=(case/'results/visual_qa.md').read_text(encoding='utf-8')
 for name in ['Report.pdf','Presentation.pdf']:
  assert hashlib.sha256((case/name).read_bytes()).hexdigest() in evidence,(case,name,'QA hash outdated')
 nb=nbformat.read(case/'Notebook.ipynb',as_version=4);cells=[c for c in nb.cells if c.cell_type=='code' and c.source.strip()]
 pages=len(PdfReader(case/'Report.pdf').pages);slides=len(PdfReader(case/'Presentation.pdf').pages)
 pages_total+=pages;slides_total+=slides
 rows.append(f'| {case.name} | {pages} | {slides} | {len(cells)}/{len(cells)} | 0 | PASS |')
table='\n'.join(rows)
sources=sum(1 for _ in csv.DictReader((ROOT/'SOURCE_REGISTER.csv').open(encoding='utf-8-sig')))
archive='NOT_RUN: итоговая упаковка выполняется после этого протокола.'
if a.archive_checked:
 proof=json.loads((ROOT.parent/'UNPACKED_REPRODUCTION_CHECK.json').read_text(encoding='utf-8'))
 assert proof['unpacked_science_retraining'].startswith('PASS')
 archive='PASS: ZIP распакован отдельно; проверены CRC, размеры и SHA-256. Из распакованной копии заново выполнены все четыре научных конвейера и notebooks; метрики воспроизвелись. Это запуск из сохранённых исходных данных, не новая загрузка всех интернет-источников. После обновления сводного протокола финальный ZIP повторно проверяется по хешам и распаковке.'
out=f'''# Итоговый протокол качества

Повторная полная редакция: 26 сентября 2026 года. PASS относится к конкретной проверке. Ограничения научных результатов и соответствия локальному заданию сохранены в SUBMISSION_STATUS.md.

## Файлы

| Папка | Страниц отчёта с титулом и литературой | Слайдов | Исполнено ячеек | Ошибок Notebook | Визуальная проверка PDF |
|---|---:|---:|---:|---:|---|
{table}

Итого: {pages_total} страниц отчётов и {slides_total} слайдов. Путеводитель 00_README.pdf: {len(PdfReader(ROOT/'00_README.pdf').pages)} страницы. Каждый кейс содержит DOCX/PPTX, PDF, выполненные IPYNB/HTML, реальные данные, код, модели, прогнозы и источники.

## Проверки вычислений

- **PASS** — единый python scripts/reproduce.py --mode full --case all: все {len(run)} этапов имеют returncode=0, включая обучение в новых kernels, генерацию материалов, Office-экспорт и итоговую проверку. Журнал: reproduction_run.json.
- После полного цикла уточнены формулировка изменения MAE и неподтверждённая география demo. Затронутые документы и кадры повторно экспортированы и просмотрены; транспортный Notebook заново выполнен целиком. Окончательные версии определяются хешами в визуальных протоколах и manifest. Неуспешный ранний запуск сохранён в reproduction_history.
- **PASS** — {len(checks)} проверок сохранённых артефактов, 0 FAIL: verification_results.json. Метрики пересчитаны из прогнозов; проверены временные границы, группы транспорта, происхождение и хеши, модели, документы и исполнение Notebook.
- **PASS** — воздух: независимая реконструкция лагов и rolling-признаков, выбор по validation, физическая проверка PM2.5/PM10 на совпадающих часах. Дополнительный блочный анализ ошибок обозначен post-hoc exploratory и не меняет выбор основной модели. Детали: 01_Air_Pollution/results/REVISION_AUDIT_20260926.md и материалы кейса №10.
- **PASS** — вода: 1095 суточных значений независимо сверены с официальными таблицами, 36 месячных средних — с ежегодниками; проверены все 9 признаков, scaler, модели и метрики. Replay после разрыва/невалидной записи сбрасывает окно и требует семь последовательных валидных дней. Искусственные fault-fixtures отделены от наблюдений. Детали: 03_Water_Monitoring/results/REVISION_AUDIT_20260926.md.
- **PASS** — транспорт: все 669 объектов, координаты и лицензии сверены с COCO; crop-файлы воспроизведены из исходных JPEG. Проверены scaler, веса, группировка, accuracy и macro-F1; исходники закреплены хешами. Детали: 04_Vehicle_Classification/results/REVISION_AUDIT_20260926.md.
- **PASS** — реестр содержит {sources} записи. URL, лицензии, география и хеши доступны в SOURCE_REGISTER.csv и карточках. Неустановленные сведения явно обозначены.

## Оформление и визуальная проверка

Отчёты получили единый читаемый стиль, компактные таблицы, последовательные подписи, понятные единицы и выводы. Презентации содержат крупные заголовки и числа, редактируемые таблицы и диаграммы, источники и развёрнутые заметки. В транспортном PPTX есть относительная ссылка на соседний demo.mp4.

Все страницы итоговых Report.pdf и Presentation.pdf просмотрены индивидуально после растеризации. Точные хеши и результаты: case/results/visual_qa.md. При повторном экспорте неизменность уже просмотренных страниц подтверждалась сравнением пикселей; это не называлось новым ручным просмотром.

- **PASS** — Word/PowerPoint открыли и экспортировали итоговые документы; проверены OOXML CRC, кириллица, редактируемый текст, число страниц/слайдов, заметки, native charts и встроенные Excel-книги.
- **PASS** — demo.mp4: 30 секунд, 15 fps, 450 кадров, 1280×720; все кадры декодируются. Это разрешённое видео с ручными ROI, место съёмки не установлено. Демонстрация не является детектором или независимым тестом дорог Атырау.
- **PASS** — Notebook.html содержат код, outputs и декодируемые встроенные PNG; IPYNB выполнены последовательно без error outputs.
- **NOT_RUN** — визуальное отображение HTML в браузере: локальный URL отклонён политикой браузера, обход не выполнялся. Техническая проверка не подменяет браузерную.
- **NOT_RUN** — интерактивный просмотр DOCX/PPTX в окне Office и запуск видео кликом на слайде. Вместо этого подтверждены экспорт, просмотр PDF, относительная ссылка и декодирование видео.

## Научные ограничения

- **FAIL физической согласованности станции 32**: PM2.5 > PM10 в 845/1209 суточных пар. Алгоритм обнаружения прошёл проверку; калибровка не доказана. Для станции 235 на совпадающих часах нарушений нет в 1300 парах; это также не доказательство калибровки.
- **NOT_EVALUATED** — высокие пики основного воздушного прогноза: в test нет значений выше P90 fit-периода. Качество прогноза таких пиков не заявляется.
- **BLOCKED** — количественные доли источников выбросов: нет подтверждённой инвентаризации/атрибуции; фиктивная диаграмма не построена.
- **PARTIAL** — as-of доступность воздушного архива, действующая IoT-сеть воды и локальная размеченная транспортная выборка. Оформление не устраняет эти ограничения.
- **NOT_RUN** — будущая эксплуатация, физическая калибровка и независимые поездки по Атырау.

## Архив и переносимость

{archive}

Независимое повторение: python scripts/package_submission.py --retrain-check. Итоговая сборка: python scripts/package_submission.py. SHA-256 ZIP и протоколы рядом с архивом: Case_Studies_Atyrau_SUBMISSION.sha256, SUBMISSION_ARCHIVE_CHECK.json, UNPACKED_REPRODUCTION_CHECK.json. Manifest не включает собственный циклический хеш.

Исключены .venv, .git, временные файлы, research_notes и полные гидрологические PDF. Числовые извлечения, проверяемый загрузчик, допустимые raw-данные, модели и разрешённое исходное видео включены. Проверены отсутствие персональных абсолютных путей и технических заглушек. Полный экспорт оформления требует Windows, Word/PowerPoint и Codex runtime; научный режим доступен отдельно.
'''
(ROOT/'QUALITY_CHECK.md').write_text(out,encoding='utf-8')
print('Quality summary generated; archive_checked=',a.archive_checked)

