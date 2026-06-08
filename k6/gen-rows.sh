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

python3 "$ROOT/k6/gen_rows.py" "$HOLDOUT" "$ROOT/k6/rows.json"
python3 "$ROOT/k6/gen_threshold.py" "$REFRAW" "$ROOT/k6/threshold"
