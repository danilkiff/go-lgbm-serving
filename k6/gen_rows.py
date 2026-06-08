"""Строит k6/rows.json из holdout.csv: матрица признаков без заголовка как
[[f0..f11], ...]. Тело POST /score в нагрузочном сценарии берёт случайный вектор
из этого списка.

    python3 gen_rows.py <holdout.csv> <rows.json>
"""

from __future__ import annotations

import csv
import json
import sys


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: gen_rows.py <holdout.csv> <rows.json>")
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as f:
        r = csv.reader(f)
        next(r)  # заголовок
        rows = [[float(x) for x in rec] for rec in r if rec]
    with open(dst, "w") as f:
        json.dump(rows, f)
    print(f"rows.json: {len(rows)} векторов x {len(rows[0])} признаков")


if __name__ == "__main__":
    main()
