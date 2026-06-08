#!/usr/bin/env bash
# Готовит вход для k6:
#   k6/rows.json  - векторы признаков из holdout.csv (без заголовка), как
#                   [[f0..f11], ...] - тело POST /score берёт случайный из них.
#   k6/threshold  - медиана raw_margin из ref_raw.csv. Отклонение при margin >
#                   threshold, поэтому медиана даёт ~50% отклонений и нагружает
#                   путь explain+GET (где только и видна разница mem vs pg).
# Запуск: k6/gen-rows.sh (из любого каталога).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOLDOUT="$ROOT/training/testdata/holdout.csv"
REFRAW="$ROOT/training/testdata/ref_raw.csv"

python3 - "$HOLDOUT" "$ROOT/k6/rows.json" <<'PY'
import csv, json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    r = csv.reader(f)
    next(r)  # заголовок
    rows = [[float(x) for x in rec] for rec in r if rec]
with open(dst, "w") as f:
    json.dump(rows, f)
print(f"rows.json: {len(rows)} векторов x {len(rows[0])} признаков")
PY

python3 - "$REFRAW" "$ROOT/k6/threshold" <<'PY'
import sys, statistics
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    next(f)  # заголовок
    vals = [float(x) for x in f if x.strip()]
med = statistics.median(vals)
with open(dst, "w") as f:
    f.write(f"{med:.6f}\n")
print(f"threshold (медиана raw_margin): {med:.6f} -> ~50% отклонений на holdout")
PY
