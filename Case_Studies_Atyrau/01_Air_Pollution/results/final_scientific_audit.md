# Финальный независимый научный аудит кейсов01/02

Проверено после завершения воздушных стадий integrated run25.09.2026 (метрики21:14, материалы21:14). Повторное обучение не запускалось. Использованы независимые формулы и сохранённые артефакты.

## PASS

- 01_Air_Pollution: metrics recalculated; saved models reproduce predictions; Report tables and editable chart values match; selected baseline matches validation; 12slides; data hash matches; local source paths portable
- 02_Air_Quality_Prediction: metrics recalculated; saved models reproduce predictions; Report tables and editable chart values match; selected baseline matches validation; 12slides; data hash matches; local source paths portable
- Station32: target/5lags/3rolling means/std/calendar independently reconstructed; latest feature date before target; metrics recalculated
- Station235: target/5lags/3rolling means/std/calendar independently reconstructed; latest feature date before target; metrics recalculated
- Station235 model parameters exactly match main pipelines; supplementary report metrics match, exploratory scope explicit
- Provenance-only rebuild left both Report.md and both slides.json byte-identical
- Global SOURCE_REGISTER currently has no excluded research_notes local_file paths

## Нерешённые ограничения

- Scientific validity remains PARTIAL: station32 PM2.5 > PM10 on845/1209paired days; calibration/channel mapping independent verification absent. Station235 consistency is more plausible, not proof of calibration.
- Operational availability remains unverified: timestamp semantic offsets, provider retrospective QC, and latency of prior-day aggregate; documentation appropriately discloses these.

## Исправленная переносимость источников

LICENSE и DATASHEET поставщика включены в01_Air_Pollution/provenance. LawHTML не распространяется; в источниках оставлен URL, без ссылки на исключённый локальный файл. build_content.py обновлён, выполнен только генератор содержимого. Хеши обоих Report.md/slides.json не изменились, повторный экспорт документов из-за этой правки не нужен.

## Итог

Не выявлено несогласованных результатов Report/slides с текущими метриками или утечки из будущего в собственных признаках. Это не снимает upstream/as-of ограничений. СтатусPARTIAL научно обоснован; не следует объявлять operational forecast или проверенную санитарную оценку.
