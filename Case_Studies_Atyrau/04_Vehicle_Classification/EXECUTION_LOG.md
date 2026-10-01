# Журнал выполнения транспортного кейса

1. Прочитано полное пользовательское задание; первичная проверка BDD100K,UA-DETRAC,COCO и лицензий через agent-reach/Jina/web. agent-reach doctor и mcporter вернули пустой вывод; Jina BDD documentation сообщила невозможность разрешить host. Использованы проверяемые первичные web-страницы.
2. HEAD официального COCO ZIP:252907541 байт. HTTPS ошибка имени сертификата; TLS не отключался. Полная HTTPзагрузка остановлена как избыточная; python scripts/acquire_annotations.py загрузил только JSON-член через Range, проверил ZIPCRC. См.annotation_download.json.
3. python scripts/prepare_data.py:337JPEG скачаны, все открываются;669 кропов подготовлены. Метки только из исходной разметки. Flickr oEmbed попытка дополнить имена авторов: сетевой timeout20 с; пропуски честно сохранены.
4. python pipeline.py:5fit, выбор C1 на validation; результаты, weights и figures сохранены. Все 12 проверок PASS. См.training_history.csv и execution.json с фактическим временем.
5. CC0WebM скачан со страницы Wikimedia; просмотрены контактные листы и отдельные кадры. python scripts/make_demo.py:30 секунд,450 кадров, два ручных ROI; отсутствует независимая video-разметка.
6. python scripts/make_artifacts.py: цифры импортированы из results, созданы Report.md,slides.json,README,DATA_CARD,словарь,источники.
7. python scripts/make_notebook.py: чистый kernel; реальный повтор pipeline без изменения параметров/test, исходный код и outputs,HTML. Статус и число ячеек в notebook_execution.json.

После разбиения не проводилось изменения фильтров,seed,классов или test для улучшения результатов. Повтор pipeline служит воспроизводимости и не расширяет подбор гиперпараметров.

Редакция 26.09.2026: исправлены недостатки проверок целостности и сохранённых результатов; исходный эксперимент не перенастраивался. Добавлены количественный разбор ошибок и slides_v2.json. Полный протокол редакции: results/REVISION_AUDIT_20260926.md.
