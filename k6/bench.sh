#!/usr/bin/env bash
# Один прогон = текущий режим EXPLAIN и нагрузка RATE по четырём perm
# (mem/pg x без лимита/1cpu+1gb). Результаты - в каталог прогона
#   k6/results/run-${EXPLAIN}-${RATE}rps/
# по два файла на perm: <perm>.json (сырой дамп k6, вкл. p99 и кастомные метрики)
# и <perm>.server.json (снимок GET /metrics scorer: queue_dropped, explained),
# плюс metadata.json с параметрами прогона. Разбор - в results/analysis.ipynb.
# Лимит 1cpu/1gb вешается только на scorer (docker-compose.limit.yml); postgres
# держит свои ресурсы.
#
#   bash k6/bench.sh                              # async, 1000rps x 30s -> run-async-1000rps
#   EXPLAIN=inline RATE=4000 bash k6/bench.sh     # inline под стрессом -> run-inline-4000rps
# Полная матрица:
#   for e in async inline; do for r in 1000 4000; do EXPLAIN=$e RATE=$r bash k6/bench.sh; done; done
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$HERE/docker-compose.yml"
LIMIT="$HERE/docker-compose.limit.yml"
PG_DSN='postgresql://scorer:scorer@postgres:5432/scorer'

export RATE="${RATE:-1000}"            # предлагаемых POST /score в секунду
export DURATION="${DURATION:-30s}"
export VUS="${VUS:-200}"
export MAX_VUS="${MAX_VUS:-800}"
export WORKERS="${WORKERS:-2}"         # воркеров explain (explain=async)
export EXPLAIN="${EXPLAIN:-async}"     # async (вне горячего пути) | inline (на горячем пути для всех)
export EXPLAIN_TRIES="${EXPLAIN_TRIES:-20}"
export THRESHOLD="$(cat "$HERE/threshold" 2>/dev/null || echo 0)"

RUN="run-${EXPLAIN}-${RATE}rps"
RUN_DIR="$HERE/results/$RUN"

[ -f "$HERE/rows.json" ] || { echo "нет k6/rows.json - сперва: bash k6/gen-rows.sh"; exit 1; }
mkdir -p "$RUN_DIR"

teardown() { docker compose -f "$BASE" -f "$LIMIT" --profile pg --profile bench down -v >/dev/null 2>&1 || true; }
trap teardown EXIT

run_perm() {
  local perm="$1" store="$2" limited="$3"
  echo; echo ">>> $perm (store=$store, limited=$limited, explain=$EXPLAIN, rate=$RATE, dur=$DURATION)"
  teardown
  export STORE="$store" PERM="$perm"
  export OUT="/results/$RUN/$perm.json"           # путь внутри k6-контейнера (./results смонтирован в /results)
  [ "$store" = "postgres" ] && export DSN="$PG_DSN" || export DSN=""

  local CF=(-f "$BASE"); [ "$limited" = "yes" ] && CF+=(-f "$LIMIT")

  # Готовность - через healthcheck'и compose (--wait), без ручных poll-циклов.
  [ "$store" = "postgres" ] && docker compose "${CF[@]}" --profile pg up -d --wait --wait-timeout 120 postgres >/dev/null
  docker compose "${CF[@]}" up -d --wait --wait-timeout 120 scorer >/dev/null
  docker compose "${CF[@]}" logs scorer 2>&1 | grep -o 'explain=[a-z]*, [0-9]* hot handles[^,]*' | head -1 | sed 's/^/    /' || true

  # k6 пишет сырой дамп в $OUT (handleSummary); серверные счётчики снимаем сами.
  docker compose "${CF[@]}" --profile bench run --rm k6 run /scripts/score-explain.js
  curl -s localhost:8080/metrics > "$RUN_DIR/$perm.server.json" 2>/dev/null || true
  echo "    server: $(cat "$RUN_DIR/$perm.server.json" 2>/dev/null)"
  teardown
}

# metadata.json прогона - поля читает ноутбук (title/explain/rate/duration/...).
cat > "$RUN_DIR/metadata.json" <<JSON
{
  "title":     "explain=${EXPLAIN}, ${RATE} rps",
  "explain":   "${EXPLAIN}",
  "rate":      ${RATE},
  "duration":  "${DURATION}",
  "workers":   ${WORKERS},
  "threshold": ${THRESHOLD},
  "host":      "$(uname -srm)",
  "cpu":       "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
}
JSON

run_perm "mem-unlimited" mem      no
run_perm "mem-limited"   mem      yes
run_perm "pg-unlimited"  postgres no
run_perm "pg-limited"    postgres yes

echo; echo "=== прогон записан: k6/results/$RUN ==="
ls -1 "$RUN_DIR"
echo "Разбор: k6/results/analysis.ipynb"
