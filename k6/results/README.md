# Результаты нагрузки

Каталог обрабатывает `analysis.ipynb`. Один прогон `bench.sh` - это каталог
`run-<explain>-<rate>rps/` (режим объяснения x предложенный RPS):

```text
run-async-4000rps/
├── metadata.json            # параметры прогона
├── mem-unlimited.json       # сырой дамп k6 (summary)
├── mem-unlimited.server.json
├── mem-limited.json
├── mem-limited.server.json
├── pg-unlimited.json
├── pg-unlimited.server.json
├── pg-limited.json
└── pg-limited.server.json
```

## Файлы perm

- `<perm>.json` - сырой `data` из k6 `handleSummary`; метрика лежит под ключом
  `metrics.<name>.values.<stat>`. Кастомные метрики сценария: `score_latency`,
  `explain_latency`, `explain_wait` (перцентили, включая `p(99)`),
  `explain_found` (`rate`), `explain_poll_misses` (`count`). Перцентили заданы
  `summaryTrendStats` в скрипте - флаг `--summary-export` p99 не отдаёт, потому и
  пишем сырой `data`.
- `<perm>.server.json` - снимок `GET /metrics` scorer на конец прогона:
  `scored`, `declined`, `queue_dropped` (потеряно объяснений), `explained`
  (сохранено; `/ duration` -> объяснений/с), `dead_lettered`.

## metadata.json

```json
{
  "title":     "explain=async, 4000 rps",
  "explain":   "async",
  "rate":      4000,
  "duration":  "30s",
  "workers":   2,
  "threshold": -3.166091,
  "host":      "Darwin 25.5.0 arm64",
  "cpu":       "Apple M4 Pro"
}
```

## Что делает ноутбук

1. Сканирует все `run-*`, читает `metadata.json` и пары `<perm>.json` +
   `<perm>.server.json`, собирает единый DataFrame.
2. perm (mem/pg x лимит) - категория внутри прогона (ось X), как алгоритм в
   референсе jwt-performance.
3. Считает производные: объяснений/с (`explained / duration`), доля потерь.
4. `plot_run(df, "run-async-4000rps")` рисует панель 2x2:
   - `/score` p99 по perm;
   - доля объяснённых отклонений (`explain_found`);
   - объяснений/с;
   - scatter: `/score` p99 vs доля найденных.

## Заметки

- Порядок perm фиксированный: `mem-unlimited`, `mem-limited`, `pg-unlimited`,
  `pg-limited`.
- Недостающие метрики -> `NaN` (влияют на сортировку).
- Лишние файлы в каталоге прогона игнорируются.
