"""Обучает бинарный классификатор LightGBM и выгружает артефакты для harness
паритета на Go.

Это эталонная сторона доказательства. Сторона Go загружает тот же самый
model.txt через C API LightGBM (cgo) и обязана воспроизвести эти числа.

Сравниваем по сырой марже (predict(raw_score=True) == C_API_PREDICT_RAW_SCORE),
а не по вероятности после сигмоиды: это сравнение выхода модели до линк-функции,
один в один. Вклады SHAP (predict(pred_contrib=True) == C_API_PREDICT_CONTRIB)
суммируются в ту же сырую маржу, что даёт стороне Go вторую, внутреннюю проверку
согласованности.

Обучение идёт с deterministic=True, force_row_wise=True, num_threads=1 -
задокументированный минимум для воспроизводимых результатов LightGBM (см. README,
"Численный паритет"). Паритет между машинами всё равно требует той же сборки и
платформы liblightgbm; скрипт пишет версию, с которой запускался, в meta.json.

Выходы (в --outdir, по умолчанию ../testdata):
  model.txt        текстовая модель LightGBM (C API: LGBM_BoosterCreateFromModelfile)
  holdout.csv      матрица признаков holdout; строка заголовка = имена признаков
  ref_raw.csv      эталонная сырая маржа, по значению на строку holdout
  ref_contrib.csv  вклады SHAP, форма (n, n_features + 1); последний столбец = базовое значение
  meta.json        версия lightgbm, параметры, seed, формы - проверяются стороной Go
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def make_synthetic(n: int, n_features: int, seed: int):
    """Несбалансированные табличные бинарные данные: форма как у фрода, без лицензий и скачиваний."""
    from sklearn.datasets import make_classification

    n_informative = max(2, n_features // 2)
    X, y = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=max(0, n_features // 5),
        weights=[0.98, 0.02],  # ~2% положительных - дисбаланс классов как у фрода
        class_sep=0.8,
        flip_y=0.01,
        random_state=seed,
    )
    names = [f"f{i}" for i in range(n_features)]
    return X.astype(np.float64), y.astype(np.int32), names


def load_csv(path: str, target: str):
    """Адаптер реального открытого датасета (например, IEEE-CIS / ULB creditcard). Только числовые столбцы."""
    import csv

    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [r for r in reader]
    if target not in header:
        raise SystemExit(f"target column {target!r} not in {header[:8]}...")
    tcol = header.index(target)
    names = [c for i, c in enumerate(header) if i != tcol]
    data = np.array(rows, dtype=np.float64)
    y = data[:, tcol].astype(np.int32)
    X = np.delete(data, tcol, axis=1)
    return X, y, names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["synthetic", "csv"], default="synthetic")
    ap.add_argument("--input", help="CSV path when --dataset csv")
    ap.add_argument("--target", default="target", help="target column name for --dataset csv")
    ap.add_argument("--n", type=int, default=20000, help="synthetic sample count")
    ap.add_argument("--features", type=int, default=30, help="synthetic feature count")
    ap.add_argument("--holdout", type=int, default=4000, help="rows held out for the parity check")
    ap.add_argument("--seed", type=int, default=708)
    ap.add_argument(
        "--outdir",
        default=str(pathlib.Path(__file__).resolve().parent.parent / "testdata"),
    )
    args = ap.parse_args()

    import lightgbm as lgb

    if args.dataset == "csv":
        if not args.input:
            raise SystemExit("--dataset csv requires --input <path>")
        X, y, names = load_csv(args.input, args.target)
    else:
        X, y, names = make_synthetic(args.n, args.features, args.seed)

    # Детерминированное разбиение holdout.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    n_hold = min(args.holdout, len(X) // 5)
    hold_idx, train_idx = perm[:n_hold], perm[n_hold:]
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_ho = X[hold_idx]

    params = {
        "objective": "binary",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_data_in_leaf": 50,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,  # bagging многопоточный и невоспроизводимый - выключаем
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "seed": args.seed,
        "verbose": -1,
    }
    booster = lgb.train(
        params,
        lgb.Dataset(X_tr, label=y_tr, feature_name=names),
        num_boost_round=200,
    )

    raw = booster.predict(X_ho, raw_score=True)  # == C_API_PREDICT_RAW_SCORE
    contrib = booster.predict(X_ho, pred_contrib=True)  # == C_API_PREDICT_CONTRIB; (n, nfeat+1)

    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "model.txt"))
    np.savetxt(out / "holdout.csv", X_ho, delimiter=",", header=",".join(names), comments="", fmt="%.10g")
    np.savetxt(out / "ref_raw.csv", raw, delimiter=",", header="raw_margin", comments="", fmt="%.17g")
    np.savetxt(
        out / "ref_contrib.csv",
        contrib,
        delimiter=",",
        header=",".join([*names, "base_value"]),
        comments="",
        fmt="%.17g",
    )
    meta = {
        "lightgbm_version": lgb.__version__,
        "dataset": args.dataset,
        "seed": args.seed,
        "n_features": len(names),
        "feature_names": names,
        "n_holdout": int(n_hold),
        "n_train": int(len(train_idx)),
        "params": params,
        "num_boost_round": 200,
        "score_is_raw_margin": True,
        "contrib_shape": list(contrib.shape),
        "positive_rate_train": float(y_tr.mean()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"lightgbm {lgb.__version__}")
    print(f"wrote {out}/ : model.txt, holdout.csv ({X_ho.shape}), ref_raw.csv, ref_contrib.csv {contrib.shape}, meta.json")
    print(f"raw margin range [{raw.min():.4f}, {raw.max():.4f}]; contrib[:, -1][0]={contrib[0, -1]:.6f} (base)")


if __name__ == "__main__":
    main()
