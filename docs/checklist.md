# Чек-лист: требования методички → реализация → доказательство

Разделы методички указаны в скобках. «Доказательство» — то, что можно открыть или запустить. Фактические результаты проверочных запусков — в [verification.json](verification.json).

## Цель и предметная область (разделы 1–2)

| Требование | Реализация | Доказательство |
|---|---|---|
| Временная составляющая, конкретный показатель | log объёма и RV на сутки D+1 | `cryptocast/features.py` (таргеты), отчёт, раздел 1 |
| Не менее двух разнородных источников | Binance REST, Fear & Greed, Hacker News, RSS, GDELT | `cryptocast/ingest/sources.py`, [data_sources.md](data_sources.md) |
| Регулярное получение новых данных | инкрементальный сбор, расписание | `python -m cryptocast run-all`, `schedule` |
| Достаточная история | свечи с 2021-01-01, ~2140 суток × 5 инструментов | вкладка «Данные», отчёт, таблица 1 |
| Пользователь и практическая задача | риск-менеджер: объём к исполнению и снижение риска | вкладка «Решение на завтра», [dashboard_questions.md](dashboard_questions.md) |

## Компонент 1. Сбор данных (раздел 3)

| Требование | Реализация | Доказательство |
|---|---|---|
| Слабоструктурированный источник | заголовки HN, RSS, GDELT | `raw.news`, `staging.news` |
| Инкрементальность | watermark в `ops.watermarks` | `tests/test_ingest.py::test_incremental_loading`, `test_fng_incremental`, `test_gdelt_two_way_watermark`; повторный run-all вставляет только новое |
| Идемпотентность | upsert `ON CONFLICT DO NOTHING` по (symbol, open_time), дате, хешу URL | `test_idempotent_two_runs_same_state`, `test_upsert_ignores_existing_keys` |
| Устойчивость к сбоям | таймауты, повторы с backoff, 429 и Retry-After, зеркала, изоляция ошибок источника | `cryptocast/ingest/http_client.py`; `test_retry_after_is_respected`, `test_backoff_then_failure_raises`, `test_failure_of_one_source_is_logged_not_raised`; GDELT со статусом partial в журнале |
| Журнал загрузок | `ops.load_log`: run_id, источник, время, статус, получено, вставлено, дубликаты, watermark до и после, ошибка | `test_load_log_records_counts`; вкладка «Данные» |
| Контроль качества, блокировка витрины | `cryptocast/quality.py`, `ops.dq_results`; код 2 при критичном сбое | `tests/test_quality_staging.py` (5 тестов сбоев); `python -m cryptocast quality` |
| Без персональных данных | авторы не читаются, подписи вырезаются | `test_news_hash_stable_and_no_authors` |

## Компонент 2. Обработка и признаки (раздел 3)

| Требование | Реализация | Доказательство |
|---|---|---|
| Единый код признаков для обучения и инференса | `build_feature_frame`, `FittedPipeline` | `test_training_and_inference_use_same_code`, `test_inference_on_truncated_equals_training_row` |
| Единый вид, пропуски, аномалии | правила в `cryptocast/staging.py` (докстринг), масштабирование 22–23 свечей, невалидные сутки, флаги аномалий по прошлому | `test_daily_rules`, `test_invalid_rows_rejected`, `test_incomplete_last_day_excluded`, `test_rv_formula` |
| Обработка текста | очистка, дедупликация, VADER, темы, упоминания, TF-IDF+SVD внутри фолда | `test_news_dedup_and_topics`, `test_transforms_fit_only_on_train` |
| Лаговые, оконные, календарные признаки, HAR | `cryptocast/features.py` | витрина `data/lake/mart/features.parquet` |
| Кодирование категорий с обоснованием | one-hot (линейные, MLP), нативная категория (LightGBM) | `test_categorical_encodings`; `python -m cryptocast analyze`; отчёт 4.5 |
| Отбор признаков и оценка влияния | top-k по gain внутри фолда, абляция: рынок / + индекс страха / + текст / + SVD | `mart/feature_analysis.parquet`; отчёт 4.6 |
| Проверка отсутствия утечки | усечение, метки времени, перемешанный таргет, преобразования на train | `python -m cryptocast check-leakage`, `mart/leakage_report.json`, `test_truncation_invariance`, `test_truncation_detects_a_leak`, `test_targets_are_next_day`, `test_scaler_ignores_test_rows` |
| Разбиение по времени, фиксированное зерно | walk-forward с зазором, отложенные 90 дней, seed 42 | `test_folds_are_ordered_and_have_gap`, `test_lightgbm_is_deterministic`, `test_mlp_is_deterministic` |

## Компонент 3. Модели и дашборд (раздел 3)

| Требование | Реализация | Доказательство |
|---|---|---|
| Не менее 3 моделей, базовая, разные классы | наивная (+ среднее 7 дн.), ElasticNet HAR, LightGBM, MLP | `cryptocast/ml/core.py`; `test_evaluation_writes_results` |
| Единая временная валидация | `run_walk_forward` для всех моделей | `mart/metrics.parquet`, `mart/folds.parquet` |
| Не менее двух метрик | MAE, RMSE, QLIKE | `cryptocast/ml/metrics.py`; `test_qlike_*` |
| Понятная пользователю величина | % от объёма, п.п. волатильности, улучшение к наивному | `test_metrics_translate_to_percent`; вкладка «Качество моделей» |
| По инструментам | метрики по каждому symbol и ALL | вкладка «Качество моделей», разрез |
| Дашборд: факт, прогноз, качество | Streamlit | `cryptocast/dashboard/app.py`; [screenshots/](screenshots/) |
| Разрезы и фильтры | инструмент, показатель, период, тип оценки | боковая панель |
| Автообновление | `st_autorefresh` каждые 300 с + кэш по времени изменения файла | `config.yaml: dashboard.autorefresh_seconds` |
| Каждый элемент отвечает на вопрос | карта вопросов | [dashboard_questions.md](dashboard_questions.md) |

## Компоненты по выбору (раздел 4)

| Требование | Реализация | Доказательство |
|---|---|---|
| A. Важность двумя способами | SHAP и permutation | `cryptocast/interpret.py`; `mart/importance_global.parquet`; вкладка «Объяснение» |
| A. Вклад признаков для отдельного прогноза | SHAP для любой пары (инструмент, дата) по модели фолда | вкладка «Объяснение» |
| A. Зависимость от признака | PDP + ICE для 8 важнейших признаков | `mart/pdp_ice.parquet` |
| A. Сравнение периодов | доли SHAP по годам, корреляция Спирмена | `mart/importance_periods.parquet`, `importance_stability.parquet` |
| B. Дрейф входных данных | PSI и KS | `cryptocast/monitoring.py`; `ops.drift_metrics`; `test_psi_*`, `test_compute_drift_flags` |
| B. Качество на фактах | скользящая MAE production-версии | `mart/quality_daily.parquet`; `test_rolling_quality` |
| B. Правило переобучения | пороги в `config.yaml: monitoring.retrain` | `test_retrain_rule_needs_consecutive_days`, `test_retrain_rule_psi`; `ops.retrain_decisions` |
| B. Реестр версий | `ml.registry`: признаки, параметры, метрики, хеш данных, эксперимент | вкладка «Мониторинг» |
| B. Вывод в эксплуатацию и откат | `promote`, `rollback`, автооткат по теневой версии | `test_promotion_and_rollback`; `python -m cryptocast rollback log_rv` |
| B. Демонстрация реакции | `simulate-drift` на копии с меткой СИМУЛЯЦИЯ | `mart/simulation_summary.json`; `test_degrade_only_after_start` |
| C. Подбор с ограничением бюджета | Optuna TPE, n_trials и timeout | `config.yaml: tuning.budgets`; `mart/optuna_trials.parquet` |
| C. Журнал экспериментов | `ml.experiments`: конфигурация, хеш config, версия данных, seed, git-коммит, метрики | вкладка «Эксперименты»; `list-experiments` |
| C. Сравнение в интерфейсе | таблица и графики экспериментов | вкладка «Эксперименты» |
| C. Отложенная выборка | последние 90 дней вне подбора | `dev_holdout` в `cryptocast/ml/experiments.py` |
| C. Воспроизведение по ID | `reproduce --experiment-id` | `test_reproduce_experiment_matches`; `artifacts/reproductions/*.json` |
| C. Детерминизм LightGBM | deterministic=True, force_row_wise=True, seed | `test_lightgbm_is_deterministic` |

## Стек и требования к нему (раздел 5)

| Требование | Реализация | Доказательство |
|---|---|---|
| Воспроизводимость развёртывания | инструкция в README, `requirements.txt` с версиями | [README.md](../README.md) |
| Автоматизация одной командой и по расписанию | `run-all`, `schedule`, планировщик Windows | README, раздел «Расписание» |
| Конфигурируемость | `config.yaml` + `.env` | `cryptocast/config.py` |
| Открытость данных | только публичные API | [data_sources.md](data_sources.md) |

## Материалы (раздел 6)

| Требование | Где |
|---|---|
| Репозиторий с историей | git, небольшие коммиты (ограничение: все созданы за один день) |
| README с архитектурой, стеком, развёртыванием | [README.md](../README.md) |
| Схема архитектуры | [architecture.mmd](architecture.mmd), [architecture.png](architecture.png) |
| Отчёт с обязательными разделами | [report.docx](report.docx) |
| Презентация с демонстрацией | [presentation.pptx](presentation.pptx) |
| Работающий прототип и офлайн-снимок | `run-all`, `dashboard`, `snapshot`, `dashboard --offline` |
| Сценарий защиты | [defense.md](defense.md) |
