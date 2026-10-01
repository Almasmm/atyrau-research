"""Create, execute in fresh kernels and export notebooks for cases 1 and 10."""
from pathlib import Path
import argparse,json,sys,time
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parent.parent

def build(case_name,number):
    c=ROOT/case_name
    def md(s):return nbformat.v4.new_markdown_cell(s)
    def code(s):return nbformat.v4.new_code_cell(s)
    nb=nbformat.v4.new_notebook()
    nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    nb.metadata.language_info={'name':'python','version':sys.version.split()[0]}
    nb.cells=[md(f'# Исходный кейс №{number}: воспроизводимый эксперимент\n\nВыполняется из чистого kernel. Источник — реальные архивные записи AirData.kz / Казгидромет. **PARTIAL:** метрология канала не подтверждена; PM2.5 часто выше PM10. Это не система санитарных предупреждений.\n\nПрочитайте DATA_CARD.md и Report.md. ИИ-помощь раскрыта на уровне пакета.'),code("from pathlib import Path\nimport sys, json\nimport pandas as pd\nfrom IPython.display import display, Image\ncase = Path.cwd()\nassert (case / 'DATA_CARD.md').exists()\nshared_code = case.parent / '01_Air_Pollution'\nsys.path.insert(0, str(shared_code))\nimport pipeline\nprint('Case:', case.name)\nprint('Pinned archive commit:', pipeline.COMMIT)"),md('## Реальное выполнение\n\nСледующая ячейка готовит данные, обучает все предусмотренные конфигурации, выбирает по validation и сохраняет test predictions. Результаты не подставлены заранее.'),code(f"summary = pipeline.run_case{1 if number==1 else 2}()\ndisplay(summary)"),md('## Данные и схема\n\nTarget никогда не интерполируется; временная сетка сохраняет реальные пропуски.'),code("dataset = pd.read_csv(case / 'dataset.csv')\nprint('Rows:', len(dataset), 'Columns:', len(dataset.columns))\ndisplay(dataset.head())\ndisplay(dataset.isna().sum().rename('missing'))"),md('## Разделение и validation'),code("display(json.loads((case/'results/splits.json').read_text(encoding='utf8')))\ndisplay(pd.read_csv(case/'results/validation_metrics.csv'))"),md('## Независимый test\n\nВсе варианты сравниваются по одинаковым target. Победитель validation не меняется по test.'),code("pred = pd.read_csv(case/'results/predictions.csv')\nmetrics = pd.read_csv(case/'results/metrics.csv')\ndisplay(metrics)\ndisplay(pred.head())\nfor row in metrics.itertuples():\n    actual = pipeline.metric(pred.actual, pred[row.model])\n    assert abs(actual['MAE']-row.MAE) < 1e-9\n    assert abs(actual['RMSE']-row.RMSE) < 1e-9\nprint('Metrics recalculated from saved predictions: PASS')")]
    if number==1:
        nb.cells += [md('## Полнота, межгородское сравнение и физическая согласованность\n\nPM2.5>PM10 — флаг для проверки паспортов и калибровки, не автоматически исправляемая ошибка.'),code("display(pd.read_csv(case/'results/coverage.csv'))\ndisplay(pd.read_csv(case/'results/city_comparison.csv'))\ndisplay(pd.read_csv(case/'results/cross_pollutant_quality.csv'))\ndisplay(pd.read_csv(case/'results/threshold_comparison.csv'))")]
        figs=['coverage.png','city_comparison.png','monthly_means.png','seasonal_heatmap.png','pollutant_exceedance.png','model_comparison.png','residuals.png']
    else:
        nb.cells += [md('## Признаки строго до target\n\nЭто собственная проверка кода. Ретроспективная очистка и реальная доступность upstream не проверяемы.'),code("date = pd.to_datetime(dataset.date)\nlatest = pd.to_datetime(dataset.latest_feature_date)\nassert (latest < date).all()\nassert not dataset.value_ugm3.isna().any()\nassert not dataset[['lag1','lag7']].isna().any().any()\nprint('Feature dates and actual-target availability: PASS')\ndisplay(pd.read_csv(case/'results/seasonal_metrics.csv'))\ndisplay(pd.read_csv(case/'results/largest_errors.csv').head())")]
        nb.cells += [md('## Вторичный exploratory-анализ станции235\n\nДобавлен после проверки физической согласованности32. Не заменяет основной test. Гиперпараметры заморожены по32; настройки по235 нет.'),code("display(pd.read_csv(case/'results/sensitivity235/metrics.csv'))\ndisplay(json.loads((case/'results/sensitivity235/metadata.json').read_text(encoding='utf8')))")]
        figs=['temporal_split.png','model_comparison.png','actual_predicted.png','residuals.png','feature_importance.png']
    nb.cells += [md('## Фактические рисунки')]
    for fig in figs:nb.cells.append(code(f"display(Image(filename=str(case / 'figures' / '{fig}')))"))
    nb.cells += [md('## Аудит и воспроизводимость'),code("display(pd.read_csv(case/'results/quality_checks.csv'))\nmetadata = json.loads((case/'models/metadata.json').read_text(encoding='utf8'))\ndisplay(metadata['versions'])\nprint('Dataset SHA-256:', metadata['dataset_sha256'])\nprint('Validation selection:', metadata['selected_on_validation_MAE'])\nprint('Source limitation:', metadata['upstream_QC_caveat'])"),md('## Вывод\n\nРеально обученные альтернативы не дали устойчивого выигрыша над заранее предусмотренным baseline. Все цифры выше получены выполненным кодом. Нужны независимая проверка канала и сведения о фактическом времени публикации измерений. Подробная интерпретация и источники — в Report.md и References.md.')]
    # Override the kernel launcher for this process only; no global/user kernel mutation.
    km=KernelManager(kernel_name='python3')
    km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    t=time.perf_counter()
    client=NotebookClient(nb,km=km,timeout=600,resources={'metadata':{'path':str(c)}},allow_errors=False)
    client.execute(cwd=str(c))
    nbformat.write(nb,c/'Notebook.ipynb')
    body,_=HTMLExporter().from_notebook_node(nb)
    (c/'Notebook.html').write_text(body,encoding='utf8')
    result={'status':'PASS','clean_kernel':True,'code_cells':sum(x.cell_type=='code' for x in nb.cells),'executed_code_cells':sum(x.cell_type=='code' and x.execution_count is not None for x in nb.cells),'error_outputs':sum(o.output_type=='error' for cell in nb.cells if cell.cell_type=='code' for o in cell.outputs),'elapsed_seconds':time.perf_counter()-t,'notebook_bytes':(c/'Notebook.ipynb').stat().st_size,'html_bytes':(c/'Notebook.html').stat().st_size}
    (c/'results'/'notebook_execution.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(case_name,result,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['1','10','both'],default='both');a=ap.parse_args()
    if a.case in ['1','both']:build('01_Air_Pollution',1)
    if a.case in ['10','both']:build('02_Air_Quality_Prediction',10)
