# Нагрузочное тестирование

Что делает:

- гоняет k6-нагрузку по REST через матрицу 2x2 (хранилище x лимит) в двух
  режимах объяснения;
- меряет латентность горячего пути `/score` и надёжность доставки объяснений
  под нагрузкой;
- складывает сырые результаты в `results/run-*` для разбора в ноутбуке.

Один прогон = режим `EXPLAIN` и нагрузка `RATE` по четырём perm; сырьё ложится в
`results/run-<explain>-<rate>rps/`, разбор и графики - в `results/analysis.ipynb`.

|            | без лимита      | 1 CPU / 1 GB    |
|------------|-----------------|-----------------|
| in-memory  | `mem-unlimited` | `mem-limited`   |
| postgresql | `pg-unlimited`  | `pg-limited`    |

Лимит 1cpu/1gb вешается только на `scorer` (`docker-compose.limit.yml`); postgres
держит свои ресурсы.

## Режимы объяснения (`scorer -explain`)

- **async** (дефолт): `/score` скорит, для отклонений кладёт событие в очередь;
  SHAP считают `-workers` фоновых воркеров вне горячего пути, результат отдаётся
  `GET /explain/{id}`. Под нагрузкой выше пропускной способности воркеров очередь
  переполняется и объяснения теряются (`queue_dropped`).
- **inline**: `/score` сам считает SHAP для КАЖДОГО решения и сохраняет объяснение
  до ответа. Ни очереди, ни воркеров - терять нечем. Цена: каждый `/score` платит
  полный SHAP.

`-store=postgres` - адаптер `pipeline.PgStore` за тем же интерфейсом `Store`, что
и `MemStore`.

## Команды

`make help` печатает список:

```sh
make data      # вход k6 из training/testdata -> testdata/ (rows.json + threshold)
make bench     # один прогон матрицы (по умолчанию async, 1000 rps)
make matrix    # полная матрица: {async,inline} x {1000,4000} rps
make notebook  # разбор результатов в jupyter
make clean     # удалить сгенерированный вход (testdata/)
```

Сперва `make -C training data` (нужны `holdout.csv`, `ref_raw.csv`), затем `make
data` здесь. Прогон под стрессом задаётся через env: `EXPLAIN=inline RATE=4000
make bench`.

Env: `EXPLAIN` (async|inline), `RATE` (POST /score в секунду), `DURATION`, `VUS`,
`MAX_VUS`, `WORKERS`, `EXPLAIN_TRIES`. Готовность сервисов - через healthcheck'и
compose (`up -d --wait`), без ручных опросов.

## Разбор результатов

`make notebook` (= `uv run jupyter notebook results/analysis.ipynb`); первый запуск
сам поднимает окружение из `uv.lock` в `.venv` (в ignore).

Раскладка `results/` и поля метрик - в `results/README.md`. Графики и выводы
(inline vs async: надёжность, пропускная способность объяснений, цена латентности)
- в самом ноутбуке.

## Раскладка

- `scripts/` - сценарий и обвязка: `score-explain.js` (k6-сценарий), `bench.sh`
  (прогон матрицы), `gen-rows.sh` (-> `gen_rows.py`, `gen_threshold.py`, готовят
  вход);
- `testdata/` - сгенерированный вход (`rows.json`, `threshold`; в ignore);
- `results/` - прогоны и ноутбук (коммитим как артефакт);
- `Dockerfile.scorer` (cgo против lib_lightgbm из колеса pip), `docker-compose.yml`,
  `docker-compose.limit.yml` - стенд; `pyproject.toml` + `uv.lock` - окружение
  ноутбука.
