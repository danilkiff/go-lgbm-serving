"""Считает k6/threshold: медиану raw_margin из ref_raw.csv. В нагрузочном
сценарии отклонение наступает при margin > threshold, поэтому медиана даёт ~50%
отклонений и нагружает путь explain+GET (где и видна разница mem vs pg).

    python3 gen_threshold.py <ref_raw.csv> <threshold>
"""

from __future__ import annotations

import statistics
import sys


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: gen_threshold.py <ref_raw.csv> <threshold>")
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as f:
        next(f)  # заголовок
        vals = [float(x) for x in f if x.strip()]
    med = statistics.median(vals)
    with open(dst, "w") as f:
        f.write(f"{med:.6f}\n")
    print(f"threshold (медиана raw_margin): {med:.6f} -> ~50% отклонений на holdout")


if __name__ == "__main__":
    main()
