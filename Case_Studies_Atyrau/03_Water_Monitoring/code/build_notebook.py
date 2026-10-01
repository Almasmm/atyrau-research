"""Create and execute an honest clean-kernel notebook and standalone HTML."""
from pathlib import Path
import json, time, os, sys
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager

CASE=Path(__file__).resolve().parents[1]

def build():
    md=nbformat.v4.new_markdown_cell; code=nbformat.v4.new_code_cell
    cells=[
      md('# Кейс №12: Урал — Атырау\n\nПрогноз среднесуточного уровня на один день. Реальные архивные наблюдения 2018, 2019, 2022. Этот Notebook выполняет обучение; результаты не вставлены вручную. Помощь ИИ раскрывается в общем пакете. Источники и ограничения: [DATA_CARD.md](DATA_CARD.md), [References.md](References.md).'),
      code("from pathlib import Path\nimport sys, json\nimport pandas as pd\nfrom IPython.display import display, Image\nCASE = Path.cwd()\nif not (CASE / 'dataset.csv').exists():\n    CASE = CASE / '03_Water_Monitoring'\nassert (CASE / 'dataset.csv').exists(), 'Запустите из папки кейса или корня пакета'\nsys.path.insert(0, str(CASE / 'code'))\nimport pipeline\nprint('Python:', sys.version.split()[0])\nprint('scikit-learn:', pipeline.sklearn.__version__)"),
      md('## Данные и контроль извлечения\n\nЕдиница — гидропост-день. Включён числовой архив, а полные PDF с неустановленной открытой лицензией не распространяются. Повтор извлечения: `python code/prepare_data.py --download --cache <папка>`.'),
      code("df = pd.read_csv(CASE / 'dataset.csv', parse_dates=['date'])\ndisplay(df.head())\ndisplay(df.groupby('source_year').agg(days=('level_cm','size'), mean_cm=('level_cm','mean'), minimum=('level_cm','min'), maximum=('level_cm','max')))\nquality = json.loads((CASE / 'results/data_quality.json').read_text(encoding='utf-8'))\nprint(json.dumps(quality, ensure_ascii=False, indent=2))\ndisplay(pd.read_csv(CASE / 'results/extraction_validation.csv'))"),
      md('## Протокол и обучение\n\nTrain — 2018; validation — 2019; test — 2022. Все измеряемые признаки сдвинуты в прошлое, лаги не пересекают отсутствующие 2020–2021. В начале 2019 доступны предыдущие дни 2018. Выбор по validation MAE. Следующая ячейка заново обучает шесть кандидатов, фиксирует выбор, переобучает два финальных метода и оценивает test.'),
      code("summary = pipeline.run()\nprint(json.dumps(summary, ensure_ascii=False, indent=2))\ndisplay(pd.read_csv(CASE / 'results/validation_search.csv'))\ndisplay(pd.read_csv(CASE / 'results/metrics.csv'))"),
      md('## EDA на фактическом ряде\n\nТри года не образуют непрерывного климатического ряда. Месячные различия не доказывают долговременный тренд.'),
      code("for name in ['01_daily_levels.png','02_monthly_levels.png','03_daily_changes.png']:\n    display(Image(filename=str(CASE / 'figures' / name), width=900))"),
      md('## Отложенная оценка и ошибки\n\nОсновной вывод определяется MAE. Высокий R² не заменяет сравнение с persistence. Ни параметры, ни состав test после оценки не изменяются.'),
      code("for name in ['05_model_comparison.png','04_actual_vs_predicted.png','06_errors_seasons.png']:\n    display(Image(filename=str(CASE / 'figures' / name), width=900))\ndisplay(pd.read_csv(CASE / 'results/largest_errors.csv'))\ndisplay(pd.read_csv(CASE / 'results/seasonal_metrics.csv'))\nprint('Изменение MAE относительно persistence, %:', round(summary['MAE_improvement_percent'], 4))"),
      md('## Исторический replay\n\nФлаг основан на 99-м процентиле прошлых абсолютных изменений. Это не порог опасности. Последний прогноз на 01.01.2023 не оценивается; истинных меток аварий нет.'),
      code("from replay import replay\nprint(json.dumps(replay(), ensure_ascii=False, indent=2))\nactions = pd.read_csv(CASE / 'results/replay_actions.csv')\ndisplay(actions.head(10))\ndisplay(actions[actions.statistical_flag])"),
      md('## Проверки и итог\n\nПересчитываются метрики из CSV, сравниваются последовательные и пакетные прогнозы, проверяются разбиение, хеши и исходные месячные средние.'),
      code("display(pd.DataFrame(pipeline.verify()))\nprint(f\"Ridge MAE: {summary['test_selected']['MAE_cm']:.3f} см; baseline: {summary['test_baseline']['MAE_cm']:.3f} см\")\nprint('Преимущество выбранной модели по test MAE не подтверждено.')"),
      md('## Ограничения\n\nОдин пост, три выбранных по доступности года; отсутствие данных 2020–2021; неизвестная оперативная задержка суточного среднего и конкретная автоматизация поста; нет внешних прогнозов погоды и меток опасных событий. Нельзя утверждать живую IoT-интеграцию, предотвращённый ущерб или фактическую экономию воды. Студент должен самостоятельно проверить и понимать эксперимент перед защитой.'),
    ]
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})
    start=time.time()
    # Pin execution to this interpreter; a global python3 kernelspec may use another environment.
    # The delivered notebook metadata stays portable and contains no machine-specific path.
    km=KernelManager(kernel_name='python3')
    km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client=NotebookClient(nb,timeout=180,km=km,resources={'metadata':{'path':str(CASE)}})
    client.execute()
    nbformat.write(nb,CASE/'Notebook.ipynb')
    html,_=HTMLExporter(embed_images=True).from_notebook_node(nb)
    (CASE/'Notebook.html').write_text(html,encoding='utf-8')
    errors=[o for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error']
    code_cells=[c for c in nb.cells if c.cell_type=='code']
    status={'status':'PASS' if not errors and all(c.execution_count for c in code_cells) else 'FAIL','clean_kernel':True,'code_cells':len(code_cells),'executed_cells':sum(c.execution_count is not None for c in code_cells),'errors':len(errors),'elapsed_seconds':time.time()-start,'html_bytes':(CASE/'Notebook.html').stat().st_size,'ipynb_bytes':(CASE/'Notebook.ipynb').stat().st_size,'images_embedded':html.count('data:image/png;base64,')}
    (CASE/'results/notebook_execution.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    assert status['status']=='PASS'
    from build_materials import build as material_build
    material_build()
    with (CASE/'QUALITY_CHECK.md').open('a',encoding='utf-8') as out:out.write(f'\nNotebook: PASS, {len(code_cells)} ячеек выполнены из чистого kernel, ошибок 0. HTML экспортирован, изображения встроены; размеры проверены. Визуальная проверка HTML выполняется отдельно.\n')
    return status

if __name__=='__main__': print(json.dumps(build(),indent=2))
