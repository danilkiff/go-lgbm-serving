#!/usr/bin/env bash
# Гоняет четыре прогона REST-нагрузки и складывает результаты в k6/results/:
#   mem-unlimited  mem-limited  pg-unlimited  pg-limited
# (in-memory/postgresql) x (без лимита / 1 CPU + 1 GB на контейнер scorer).
# Лимит вешается только на scorer (docker-compose.limit.yml); postgres держит
# свои ресурсы. Параметры нагрузки - через окружение (см. значения ниже).
#
#   bash k6/bench.sh                 # все четыре прогона с дефолтами
#   DURATION=10s RATE=300 bash k6/bench.sh   # короче/легче
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$HERE/docker-compose.yml"
LIMIT="$HERE/docker-compose.limit.yml"
RESULTS="$HERE/results"
PG_DSN='postgresql://scorer:scorer@postgres:5432/scorer'

export RATE="${RATE:-1000}"            # предлагаемых POST /score в секунду
export DURATION="${DURATION:-30s}"
export VUS="${VUS:-200}"
export MAX_VUS="${MAX_VUS:-800}"
export WORKERS="${WORKERS:-2}"         # воркеров explain (explain=async)
export EXPLAIN="${EXPLAIN:-async}"     # async (вне горячего пути) | inline (на горячем пути для всех)
export EXPLAIN_TRIES="${EXPLAIN_TRIES:-20}"
export THRESHOLD="$(cat "$HERE/threshold" 2>/dev/null || echo 0)"

[ -f "$HERE/rows.json" ] || { echo "нет k6/rows.json - сперва: bash k6/gen-rows.sh"; exit 1; }
mkdir -p "$RESULTS"

teardown() { docker compose -f "$BASE" -f "$LIMIT" --profile pg --profile bench down -v >/dev/null 2>&1 || true; }
trap teardown EXIT

wait_scorer() { for i in $(seq 1 300); do curl -sf -o /dev/null localhost:8080/metrics && return 0; done; return 1; }

run_perm() {
  local perm="$1" store="$2" limited="$3"
  echo; echo ">>> $perm (store=$store, limited=$limited, rate=$RATE, dur=$DURATION)"
  teardown
  export STORE="$store" PERM="$perm"
  [ "$store" = "postgres" ] && export DSN="$PG_DSN" || export DSN=""

  local CF=(-f "$BASE"); [ "$limited" = "yes" ] && CF+=(-f "$LIMIT")

  if [ "$store" = "postgres" ]; then
    docker compose "${CF[@]}" --profile pg up -d postgres >/dev/null
    local hc=""; for i in $(seq 1 120); do hc=$(docker inspect -f '{{.State.Health.Status}}' lgbm-bench-postgres-1 2>/dev/null || true); [ "$hc" = "healthy" ] && break; done
    echo "    pg health=$hc"
  fi

  docker compose "${CF[@]}" up -d scorer >/dev/null
  wait_scorer || { echo "    scorer не поднялся:"; docker compose "${CF[@]}" logs scorer | tail -20; return 1; }
  docker compose "${CF[@]}" logs scorer 2>&1 | grep -o 'explain=[a-z]*, [0-9]* hot handles[^,]*' | head -1 | sed 's/^/    /'

  # k6 в фоне; пока он крутит нагрузку, снимаем CPU/MEM контейнеров - это и есть
  # аргумент к bottleneck (CPU у scorer vs занятость postgres).
  docker compose "${CF[@]}" --profile bench run --rm k6 run /scripts/score-explain.js &
  local k6pid=$!
  : > "$RESULTS/$perm.stats.txt"
  while kill -0 "$k6pid" 2>/dev/null; do
    docker stats --no-stream --format '{{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}}' 2>/dev/null \
      | grep -E 'lgbm-bench-(scorer|postgres)' >> "$RESULTS/$perm.stats.txt" || true
  done
  wait "$k6pid" 2>/dev/null || true
  echo "    peak cpu: $(sort -t= -k2 -rn "$RESULTS/$perm.stats.txt" 2>/dev/null | grep scorer | head -1)"

  curl -s localhost:8080/metrics > "$RESULTS/$perm.metrics.json" 2>/dev/null || true
  echo "    server: $(cat "$RESULTS/$perm.metrics.json" 2>/dev/null)"
  teardown
}

run_perm "mem-unlimited" mem      no
run_perm "mem-limited"   mem      yes
run_perm "pg-unlimited"  postgres no
run_perm "pg-limited"    postgres yes

echo; echo "=== сводка (k6/results) ==="
python3 - "$RESULTS" <<'PY'
import json, glob, os, sys
res = sys.argv[1]
order = ["mem-unlimited", "mem-limited", "pg-unlimited", "pg-limited"]
rows = {}
for p in glob.glob(os.path.join(res, "*.json")):
    if p.endswith(".metrics.json"): continue
    try:
        d = json.load(open(p)); rows[d["perm"]] = d
    except Exception:
        pass
def g(d, *ks):
    cur = d
    for k in ks:
        cur = cur.get(k) if isinstance(cur, dict) else None
        if cur is None: return "-"
    return cur
hdr = f'{"perm":<15}{"score p99":>11}{"GET p99":>10}{"wait p99":>10}{"found":>7}{"req/s":>9}'
print(hdr); print("-" * len(hdr))
for k in order:
    d = rows.get(k)
    if not d: continue
    print(f'{k:<15}{str(g(d,"score_ms","p99")):>11}{str(g(d,"explain_get_ms","p99")):>10}'
          f'{str(g(d,"explain_wait_ms","p99")):>10}{str(g(d,"explain_found_rate")):>7}{str(g(d,"http_req_rate")):>9}')
PY
echo "(JSON по прогонам: k6/results/<perm>.json, снимки сервера: <perm>.metrics.json)"
