# Нагрузочное тестирование

k6-нагрузка через REST по матрице 2x2 (хранилище x лимит) в двух режимах
объяснения. Один прогон `bench.sh` = режим `EXPLAIN` и нагрузка `RATE` по четырём
perm; сырые результаты ложатся в `results/run-<explain>-<rate>rps/`, разбор и
графики - в `results/analysis.ipynb`.

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

```sh
bash k6/gen-rows.sh                          # rows.json + threshold (~50% отклонений)
EXPLAIN=async  RATE=1000 bash k6/bench.sh     # один прогон -> results/run-async-1000rps/
EXPLAIN=inline RATE=4000 bash k6/bench.sh     # inline под стрессом
# полная матрица:
for e in async inline; do for r in 1000 4000; do EXPLAIN=$e RATE=$r bash k6/bench.sh; done; done
```

Env: `EXPLAIN` (async|inline), `RATE` (POST /score в секунду), `DURATION`, `VUS`,
`MAX_VUS`, `WORKERS`, `EXPLAIN_TRIES`. Готовность сервисов - через healthcheck'и
compose (`up -d --wait`), без ручных опросов.

## Разбор результатов

```sh
uv run --project k6 jupyter notebook k6/results/analysis.ipynb
```

Первый `uv run` сам поднимает окружение из `k6/uv.lock` в `k6/.venv` (в ignore).

Раскладка `results/` и поля метрик - в `results/README.md`. Графики и выводы
(inline vs async: надёжность, пропускная способность объяснений, цена латентности)
- в самом ноутбуке.

## Файлы

`Dockerfile.scorer` (cgo против lib_lightgbm из колеса pip), `docker-compose.yml`,
`docker-compose.limit.yml`, `score-explain.js` (k6-сценарий), `bench.sh`,
`gen-rows.sh` (-> `gen_rows.py`, `gen_threshold.py`), `pyproject.toml` + `uv.lock`
(окружение ноутбука).
