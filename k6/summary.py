"""Сводит результаты прогонов bench.sh из <results>/<perm>.json в таблицу
(score/GET/wait p99, доля найденных объяснений, req/s) по четырём перестановкам.
Снимки сервера <perm>.metrics.json пропускаются.

    python3 summary.py <results-dir>
"""

from __future__ import annotations

import glob
import json
import os
import sys


def g(d, *ks):
    cur = d
    for k in ks:
        cur = cur.get(k) if isinstance(cur, dict) else None
        if cur is None:
            return "-"
    return cur


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summary.py <results-dir>")
    res = sys.argv[1]
    order = ["mem-unlimited", "mem-limited", "pg-unlimited", "pg-limited"]
    rows = {}
    for p in glob.glob(os.path.join(res, "*.json")):
        if p.endswith(".metrics.json"):
            continue
        try:
            d = json.load(open(p))
            rows[d["perm"]] = d
        except Exception:
            pass
    hdr = f'{"perm":<15}{"score p99":>11}{"GET p99":>10}{"wait p99":>10}{"found":>7}{"req/s":>9}'
    print(hdr)
    print("-" * len(hdr))
    for k in order:
        d = rows.get(k)
        if not d:
            continue
        print(f'{k:<15}{str(g(d,"score_ms","p99")):>11}{str(g(d,"explain_get_ms","p99")):>10}'
              f'{str(g(d,"explain_wait_ms","p99")):>10}{str(g(d,"explain_found_rate")):>7}{str(g(d,"http_req_rate")):>9}')


if __name__ == "__main__":
    main()
