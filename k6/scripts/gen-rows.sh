#!/usr/bin/env bash
# Готовит вход для k6 в k6/testdata/:
#   rows.json  - векторы признаков из holdout.csv (без заголовка), как
#                [[f0..f11], ...] - тело POST /score берёт случайный из них.
#   threshold  - медиана raw_margin из ref_raw.csv. Отклонение при margin >
#                threshold, поэтому медиана даёт ~50% отклонений и нагружает
#                путь explain+GET (где только и видна разница mem vs pg).
# Запуск: make -C k6 data (из любого каталога).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # k6/scripts
K6="$(cd "$HERE/.." && pwd)"                            # k6
ROOT="$(cd "$K6/.." && pwd)"                            # корень репозитория
HOLDOUT="$ROOT/training/testdata/holdout.csv"
REFRAW="$ROOT/training/testdata/ref_raw.csv"

[ -f "$HOLDOUT" ] && [ -f "$REFRAW" ] || { echo "нет эталонов в training/testdata - сперва: make -C training data"; exit 1; }
mkdir -p "$K6/testdata"
python3 "$HERE/gen_rows.py" "$HOLDOUT" "$K6/testdata/rows.json"
python3 "$HERE/gen_threshold.py" "$REFRAW" "$K6/testdata/threshold"
