"""Create overview text and a 2-page PDF directly from saved experiment results."""
from pathlib import Path
import json,csv,platform,hashlib
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def main():
 a1=read('01_Air_Pollution/results/summary.json');a2=read('02_Air_Quality_Prediction/results/summary.json')
 m2=pd.read_csv(ROOT/'02_Air_Quality_Prediction/results/metrics.csv').set_index('model')
 w=read('03_Water_Monitoring/results/summary.json');v=read('04_Vehicle_Classification/results/metrics.json')
 vw=v['test'][v['selected_model']]
 results=[
  ('1','Архив качества воздуха',f"{a1['common_days']} общих суток трёх станций; PM2.5 > PM10 в {a1['atyrau_pm25_gt_pm10_days']}/{a1['atyrau_paired_days']} пар Атырау",'PARTIAL','Нужна проверка каналов и калибровки. Нет измеренных долей источников выбросов.'),
  ('10','Прогноз качества воздуха',f"Persistence: MAE {m2.loc['persistence','MAE']:.3f}, RMSE {m2.loc['persistence','RMSE']:.3f} мкг/м³; test n={int(m2.loc['persistence','n'])}",'PARTIAL','ML не улучшила baseline. Офлайн-архив с ретроспективным контролем качества.'),
  ('12','Уровень Урала у Атырау',f"Ridge: MAE {w['test_selected']['MAE_cm']:.3f} см; persistence {w['test_baseline']['MAE_cm']:.3f} см; MAE выше на {w['test_selected']['MAE_cm']-w['test_baseline']['MAE_cm']:.3f} см",'PARTIAL','Прогноз и replay выполнены. Действующая автоматическая IoT-сеть не подтверждена.'),
  ('15','Классификация транспорта',f"Logistic: macro-F1 {vw['macro_f1']:.3f}, accuracy {vw['accuracy']:.3f}; test n={vw['n']}",'PARTIAL','Основной test — фотографии COCO. Нет размеченного видео Атырау; место съёмки demo не установлено.')]
 status='# Итоговый статус\n\nВсе четыре эмпирических эксперимента проведены. Общий статус **PARTIAL**: перечисленные ограничения соответствия заданию не устранены оформлением файлов. READY_FOR_REVIEW для всего проекта не заявляется.\n\n'
 status+='| Исходный кейс | Данные и метод | Фактический результат | Статус | Недостающие условия |\n|---|---|---|---|---|\n'
 status+='\n'.join('| '+' | '.join(x)+' |' for x in results)
 status+='\n\nКомплект каждого кейса: Report.docx/PDF, Presentation.pptx/PDF, выполненный Notebook.ipynb/HTML, dataset.csv, словарь и карточка данных, код, figures, results, обученные модели и источники. Фактическое наличие, количество страниц и проверки приведены в QUALITY_CHECK.md.\n\nФИО студента, группа и состав группы не установлены. Задание не отправлялось. Пользователь подтвердил, что PDF из assignment 2 относится к другому заданию. Перед защитой необходим самостоятельный разбор и соблюдение раскрытия ИИ-помощи.\n'
 (ROOT/'SUBMISSION_STATUS.md').write_text(status,encoding='utf-8')
 readme='''# Четыре исследования с фокусом на Атырау

Откройте `00_README.pdf`, затем `SUBMISSION_STATUS.md`. Основные отчёты и презентации находятся в четырёх папках, соответствующих исходным темам №1, №10, №12 и №15. Два воздушных кейса используют общий источник и разные производные таблицы. Это не четыре независимых набора данных.

Материалы подготовлены с ИИ-помощью, раскрытой в `AI_USAGE_DISCLOSURE.md`. Пользователь подтвердил, что четырём кейсам не относится соседний PDF с отдельным условием no AI. Это не подтверждение оценки и не замена самостоятельной защиты.

## Открытие и структура

- `Report.pdf` — версия для чтения, `Report.docx` — редактируемый текст.
- `Presentation.pdf` — просмотр, `Presentation.pptx` — 15 слайдов: редактируемый текст, таблицы, графики и заметки для выступления.
- `Notebook.html` — код и фактические выводы без Python. `Notebook.ipynb` — выполненный блокнот.
- `results/` — метрики, прогнозы, разбиения, журналы и проверки. `models/` — реально обученные модели.
- `DATA_CARD.md`, `data_dictionary.csv`, `SOURCE_REGISTER.csv` — происхождение и ограничения.
- `DEFENSE_GUIDE_RU.md` — объяснения и вопросы для самостоятельного разбора.
- `QUALITY_CHECK.md` — что проверено фактически. `SUBMISSION_MANIFEST.csv` — размеры и SHA-256 файлов.

## Воспроизведение

Проверенная среда: Windows 11, Python 3.12.14, CPU. Точные версии в `requirements.txt` и `environment.json`. Установки выполнены в `.venv` рядом с пакетом с чтением библиотек поставляемого Codex runtime. Системный Python не изменён. В чистой среде установите зависимости из файла.

```bash
python -m venv .venv
# Windows: .venv/Scripts/activate; Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --prefix .venv --name atyrau
python scripts/reproduce.py --mode verify
python scripts/reproduce.py --mode science --case all
python scripts/reproduce.py --mode full --case all
```

`verify` проверяет сохранённые результаты без обучения. `science` повторяет существенную подготовку/обучение/оценку, notebooks и текстовые материалы. `full` также пересоздаёт DOCX/PPTX/PDF и выполняет проверки пакета. `--case 01`, `02`, `03`, `04` ограничивает кейс; воздух №10 зависит от подготовленных данных №1. Точные команды также перечислены в case README. На Windows предпочтителен Python-вход, bash нужен только для оболочки reproduce.sh.

Для пересоздания PPTX нужен поставляемый Codex пакет `@oai/artifact-tool` 2.8.59 и Node, а не публичная npm-установка. Укажите `ARTIFACT_TOOL_ENTRY` на `dist/artifact_tool.mjs` этого пакета, `NODE_EXE` на Node. Для окончательной проверки по навыку задайте `PRESENTATION_SKILL_DIR` и `RUNTIME_PYTHON`. Скрипт на этом Windows автоматически определяет стандартное расположение runtime относительно профиля пользователя. PDF экспортируется локальным Microsoft Word/PowerPoint через COM без публикации. На системе без Office полный экспорт PDF нужно выполнить в доступной среде Office; автономный Linux-экспорт не заявлен.

Экспорт Office запускается через PowerShell 7 (`POWERSHELL_EXE` для другого расположения). Старый Windows PowerShell 5.1 зависал в проверенной среде и не используется. Научный режим транспорта также требует ffmpeg с H.264 в PATH.

Не требуются API-ключи для выбранных загрузок. Потребуется интернет, если исходные архивы отсутствуют. Полные официальные гидрологические ежегодники не распространяются в ZIP: скрипт скачивает их по опубликованным URL, проверяет зафиксированные SHA-256 и сверяет таблицы. Происхождение видеозаписи и условия отдельных изображений перечислены в транспортном кейсе. Учитывайте некоммерческие и ShareAlike условия лицензий, где они указаны. Хеши позволяют отличить исходный снимок данных от обновлённого файла.

## Проверка переносимости

Для проверки всех четырёх экспериментов после распаковки: `python scripts/package_submission.py --retrain-check`. Процесс создаёт отдельную копию, повторяет научные конвейеры и сверяет метрики. Фактический результат записывается в `UNPACKED_REPRODUCTION_CHECK.json` рядом с ZIP. Обновления этой редакции описаны в `REVISION_LOG_20260926.md`.

## Что ещё нужно

Провайдеру воздуха требуется подтвердить каналы/калибровку и временные отметки. Для полного локального транспортного кейса нужны разрешённое видео Атырау, независимые поездки и проверенная разметка. Водный кейс показывает историческую поддержку решений, без подключённых датчиков и исполнительных устройств. Состав группы, реквизиты студента и календарный срок следует проверить перед сдачей.

'''
 readme+='## Реальные результаты\n\n'+'\n'.join('- №'+x[0]+': '+x[2]+'. '+x[4] for x in results)+'\n'
 (ROOT/'README.md').write_text(readme,encoding='utf-8')
 fonts=[Path('C:/Windows/Fonts/arial.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
 font=next(p for p in fonts if p.exists());pdfmetrics.registerFont(TTFont('Report',str(font)))
 styles=getSampleStyleSheet()
 for style in styles.byName.values():style.fontName='Report'
 styles['Normal'].fontSize=10;styles['Normal'].leading=14;styles['Normal'].spaceAfter=8
 styles['Title'].fontSize=23;styles['Title'].leading=28;styles['Title'].alignment=TA_LEFT
 styles['Heading2'].fontSize=14;styles['Heading2'].leading=19
 P=lambda s:Paragraph(escape(s),styles['Normal'])
 story=[Paragraph('Четыре исследования с фокусом на Атырау',styles['Title']),P('Искусственный интеллект и машинное обучение • 26 сентября 2026 года — повторная проверка'),P('Исходные темы №1, №10, №12 и №15. Преподаватель: Zhanar Oralbekova. ФИО студента и группа не установлены.'),Paragraph('Результаты и статус',styles['Heading2'])]
 for n,title,result,status_,lim in results:
  story += [Paragraph('№'+n+' '+title,styles['Heading2']),P(result),P(status_+'. '+lim)]
 story += [PageBreak(),Paragraph('Как пользоваться пакетом',styles['Title']),P('Сначала прочитайте SUBMISSION_STATUS.md и ограничения. Затем откройте Report.pdf и Presentation.pdf интересующего кейса. Редактируемые файлы находятся рядом. Notebook.html позволяет увидеть реальный код и сохранённые выводы без установки Python.'),P('В каждой папке: данные или индекс изображений, карточка данных, словарь, код, графики, метрики, прогнозы, разбиения и модели. Два кейса по воздуху используют общий архив; прогноз отделён от описательного анализа.'),Paragraph('Проверка и повторение',styles['Heading2']),P('Быстрый запуск: python scripts/reproduce.py --mode verify. Повторение экспериментов: python scripts/reproduce.py --mode science --case all. Полный запуск с документами: python scripts/reproduce.py --mode full --case all. Среда и ограничения экспорта описаны в README.md и ENVIRONMENT.md.'),P('Полные гидрологические PDF в ZIP не включены. Для восстановления исходников используется загрузчик с точными URL и хешами. Для полного экспорта документов требуется Windows с Office и Codex runtime. Проверки архива и фактическое число страниц находятся в QUALITY_CHECK.md.'),Paragraph('Перед защитой',styles['Heading2']),P('Пакет подготовлен с существенной ИИ-помощью. Необходимо самостоятельно разобрать код, источники и отрицательные результаты, проверить реквизиты и срок. Материалы не были отправлены преподавателю. Процент оригинальности и итоговая оценка не заявлены.'),P('Общий статус PARTIAL. Для закрытия ограничений нужны проверка воздушных каналов, локальная размеченная транспортная выборка и подтверждение датчиков для полноценной IoT-интеграции. DEFENSE_GUIDE_RU.md объясняет реальные методы и слабые места.')]
 SimpleDocTemplate(str(ROOT/'00_README.pdf'),pagesize=A4,leftMargin=45,rightMargin=45,topMargin=40,bottomMargin=40).build(story)
 print('overview files written')
if __name__=='__main__':main()
