"""Create and execute a clean-kernel notebook, then export HTML with embedded plots."""
from pathlib import Path
import sys,json,time
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
ROOT=Path(__file__).resolve().parents[1]

def main():
    nb=nbformat.v4.new_notebook()
    cells=[]
    def md(s):cells.append(nbformat.v4.new_markdown_cell(s))
    def code(s):cells.append(nbformat.v4.new_code_cell(s))
    md('# Кейс №15: реальные кропы COCO\n\nВыполняемый эксперимент. Видеорегистратор Атырау не оценён; статус PARTIAL. Все модели обучаются с нуля. Источники и ограничения в DATA_CARD.md. Все ячейки выполняются последовательно в чистом kernel.')
    code("from pathlib import Path\nimport json, sys, importlib\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT=Path.cwd()\nassert (ROOT/'pipeline.py').exists()\nprint('Python:',sys.version.split()[0])\nprint('Source metadata:')\ndisplay(pd.Series(json.loads((ROOT/'results/preparation.json').read_text(encoding='utf-8'))))")
    md('## План и реальное обучение\nTarget: car,bus,truck. Baseline: большинство train. Метрика выбора: validation macro-F1. Фиксированный grouped split; test не выбирает гиперпараметры. Признаки HOG, RGB-гистограммы и отношение сторон; scaler только train. Следующая ячейка сама создаёт детерминированное разбиение, если индекс только что подготовлен, затем извлекает1813признаков и выполняет5fit. Предварительный запуск pipeline вне Notebook не требуется. Повтор не изменяет параметры поиска, seed или test; outputs вычисляются реально.')
    code("sys.path.insert(0,str(ROOT))\nimport pipeline\npipeline.main()")
    md('## Данные и структура\nРамки и классы исходные COCO; predicted labels не используются как эталон. Отбор задаётся лицензией, crowd=0 и минимумом40пикселей. Изображения raw и обработанные кропы разделены. Разбиение создано предыдущей исполняемой ячейкой до обучения.')
    code("data=pd.read_csv(ROOT/'dataset.csv')\ndisplay(data[['object_id','image_id','class','bbox_width','bbox_height','license_id','split']].head(8))\ndisplay(pd.crosstab(data['class'],data['split']))\nprint('Images:',data.image_id.nunique(),'objects:',len(data),'required missing:',data[['object_id','class','crop_path','split']].isna().sum().sum())")
    md('## Аудит разбиения\nПротокол зафиксирован pipeline до fit. Все объекты исходной фотографии и её группы остаются в одной части; точные кропы между частями не пересекаются.')
    code("display(pd.Series(json.loads((ROOT/'results/split_protocol.json').read_text(encoding='utf-8'))))\nassert data.groupby('image_id')['split'].nunique().max()==1\nassert data.groupby('scene_group')['split'].nunique().max()==1\nassert data.groupby('crop_sha256')['split'].nunique().max()==1\nprint('Split checks passed')")
    md('## История обучения и сравнение\nНе скрывайте разницу train/validation/test. Выбор — по validation macro-F1; accuracy здесь лишь дополнительная характеристика.')
    code("display(pd.read_csv(ROOT/'results/training_history.csv'))\ndisplay(pd.read_csv(ROOT/'results/metrics.csv'))\nmetrics=json.loads((ROOT/'results/metrics.json').read_text(encoding='utf-8'))\nprint('Selected:',metrics['selected_model'])")
    code("display(Image(filename=str(ROOT/'figures/class_distribution.png')))\ndisplay(Image(filename=str(ROOT/'figures/aspect_ratios.png')))\ndisplay(Image(filename=str(ROOT/'figures/model_comparison.png')))")
    md('## Ошибки по классам и количественный аудит')
    code("display(pd.read_csv(ROOT/'results/classification_report.csv'))\ndisplay(pd.read_csv(ROOT/'results/confusion_matrix.csv'))\ndisplay(Image(filename=str(ROOT/'figures/confusion_matrix.png')))\ndisplay(Image(filename=str(ROOT/'figures/error_examples.png')))")
    code("pred=pd.read_csv(ROOT/'results/test_predictions.csv')\nfrom sklearn.metrics import f1_score,accuracy_score\nfor name in ['majority','logistic','random_forest']:\n    actual=f1_score(pred.actual,pred[name],labels=metrics['classes'],average='macro',zero_division=0)\n    assert abs(actual-metrics['test'][name]['macro_f1'])<1e-12\n    print(name, 'recomputed macro-F1',actual,'accuracy',accuracy_score(pred.actual,pred[name]))\nprint('Group bootstrap:')\ndisplay(pd.Series(json.loads((ROOT/'results/bootstrap.json').read_text(encoding='utf-8'))))")
    md('## Демонстрация отдельно от теста\n30секунд реального CC0-видео Fernost; ручные кропы имеют свои timestamp. Нет детектора или локальной количественной оценки. Сохранённая модель даёт только предсказанный класс; он не является независимой эталонной меткой.')
    code("from scripts.make_demo import main as make_demo\nmake_demo()\ndisplay(pd.Series(json.loads((ROOT/'results/demo_metadata.json').read_text(encoding='utf-8'))))\ndisplay(Image(filename=str(ROOT/'figures/demo_frame.png')))")
    md('## Вывод и проверка\nMacro-F1 улучшился относительно majority, но accuracy выбранной модели совпала с baseline. Truck остаётся слабым классом. Сильный train/test разрыв указывает на переобучение. Для исходного локального кейса нужны разрешённые независимые размеченные поездки Атырау и оценка detector. Источники: References.md; атрибуция изображений: results/image_attribution.csv.')
    code("checks=pipeline.verify()\ndisplay(pd.Series(checks,name='PASS'))\nassert all(checks.values())")
    nb.cells=cells;nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':sys.version.split()[0]}}
    nbformat.write(nb,ROOT/'Notebook.ipynb')
    # Use the active executable without baking author-specific paths into delivered notebook.
    from jupyter_client import KernelManager
    km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client=NotebookClient(nb,timeout=600,km=km,resources={'metadata':{'path':str(ROOT)}})
    start=time.perf_counter();client.execute();nbformat.write(nb,ROOT/'Notebook.ipynb')
    html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(ROOT/'Notebook.html').write_text(html,encoding='utf-8')
    codecells=[c for c in nb.cells if c.cell_type=='code'];err=[o for c in codecells for o in c.outputs if o.output_type=='error']
    assert not err and all(c.execution_count is not None for c in codecells)
    meta={'status':'PASS','code_cells':len(codecells),'executed_code_cells':len(codecells),'errors':0,'elapsed_seconds':time.perf_counter()-start,'html_bytes':(ROOT/'Notebook.html').stat().st_size,'fresh_kernel':True,'pipeline_retrained':True}
    (ROOT/'results/notebook_execution.json').write_text(json.dumps(meta,indent=2));print(meta)

if __name__=='__main__':main()
