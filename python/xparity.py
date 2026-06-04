"""Сравнивает два кросс-платформенных дампа предсказаний (из cmd/dump) и сообщает расхождение.

Оба файла должны идти от одних model.txt и holdout.csv, посчитанных подачей Go на
разных платформах. Любое ненулевое различие - кросс-платформенное численное
расхождение одной и той же модели, эффект, который LightGBM документирует для
разных ОС, компиляторов и архитектур CPU (см. README, "Численный паритет").

    uv run python xparity.py <a.csv> <b.csv>
"""

from __future__ import annotations

import sys

import numpy as np


def load(path: str):
    a = np.loadtxt(path, delimiter=",", skiprows=1)
    return a[:, 0], a[:, 1:]  # сырая маржа, вклады (признаки... + база)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: xparity.py <a.csv> <b.csv>")
    ra, ca = load(sys.argv[1])
    rb, cb = load(sys.argv[2])
    if ra.shape != rb.shape or ca.shape != cb.shape:
        raise SystemExit("shape mismatch between the two dumps")

    n = len(ra)
    nfeat = ca.shape[1] - 1  # последний столбец вкладов - базовое (ожидаемое) значение
    raw_d = np.abs(ra - rb)
    flips = int(np.sum((ra > 0) != (rb > 0)))
    c_d = np.abs(ca - cb)

    k = 3
    top_a = np.argsort(-np.abs(ca[:, :nfeat]), axis=1)[:, :k]
    top_b = np.argsort(-np.abs(cb[:, :nfeat]), axis=1)[:, :k]
    top_mismatch = int(np.sum(np.any(top_a != top_b, axis=1)))

    print(f"rows={n}  features={nfeat}")
    print(f"[raw margin]   maxD={raw_d.max():.3e}  meanD={raw_d.mean():.3e}  decision flips={flips}")
    print(f"[SHAP]         maxD={c_d.max():.3e}  meanD={c_d.mean():.3e}")
    print(f"[reason codes] top-{k} ordering mismatches={top_mismatch}/{n}")
    print(
        f"raw d>0: {int(np.sum(raw_d > 0))}/{n}   "
        f"d>1e-9: {int(np.sum(raw_d > 1e-9))}   "
        f"d>1e-6: {int(np.sum(raw_d > 1e-6))}   "
        f"d>1e-3: {int(np.sum(raw_d > 1e-3))}"
    )


if __name__ == "__main__":
    main()
