# Фактически выполненные проверки

| check | status |
| --- | --- |
| schema_unique_dates_finite_numeric | PASS |
| strictly_past_observations | PASS |
| target_split_disjoint | PASS |
| metrics_recomputed_from_saved_predictions | PASS |
| 36_source_monthly_mean_checks | PASS |
| model_dataset_hash | PASS |
| sequential_replay_matches_batch_predictions | PASS |

Визуальная проверка исходных таблиц: PASS (EXTRACTION_QA.md). Отчётные страницы и слайды проверяются общим сборщиком; до отдельного подтверждения не объявляются проверенными. Выполнение Notebook и HTML фиксирует notebook_execution.json после выполнения.

Notebook: PASS; см. results/notebook_execution.json. Визуальная проверка HTML выполняется отдельно.
