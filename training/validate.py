"""Валидация качества RBA-модели: считает и пишет results/validation.json.

Артефакт подтверждает данными утверждения README/DESIGN о качестве - явно и
воспроизводимо:

  1. Качество выгруженной модели (serving/fixtures/model.txt) на том же
     детерминированном holdout, что и при обучении: ROC-AUC, PR-AUC, lift,
     рабочая точка margin>0, кривые ROC/PR. При --testdata дополнительно
     сверяется выравнивание строк с holdout.csv и предсказания с ref_raw.csv.
  2. Распределение gain по признакам выгруженной модели ("ни один признак не
     доминирует").
  3. Случайный против временного сплита (модели переобучаются на подвыборках с
     теми же TRAIN_PARAMS): дрейф и режим "предсказываем будущее". PR-AUC между
     сплитами несравнима (растёт с base rate) - сравнивается lift = PR-AUC/base.
  4. Контраст утечки: + глобальные частоты страны и ASN - признаки, выброшенные
     в DESIGN ("Данные и признаки"), и почему.

Тяжёлый шаг (полный датасет ~31M строк, переобучения) выполняется здесь один
раз; results/analysis.ipynb только рендерит готовый JSON. Запуск:

    make -C training validate       # нужен testdata/rba-dataset.csv (make data-rba)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from train import TRAIN_PARAMS, load_rba


def downsample(*arrays: np.ndarray, n: int = 257) -> list[list[float]]:
    """Прореживает кривые до n равноотстоящих точек (с концами) для компактного JSON."""
    m = len(arrays[0])
    idx = np.unique(np.linspace(0, m - 1, min(n, m)).astype(int))
    return [np.round(a[idx], 6).tolist() for a in arrays]


def eval_scores(y: np.ndarray, s: np.ndarray) -> dict:
    """Метрики ранжирования и lift для одного теста. lift = PR-AUC / base rate:
    в отличие от абсолютной PR-AUC сравним между тестами с разной долей атак."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    base = float(y.mean())
    pr = float(average_precision_score(y, s))
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": round(base, 6),
        "roc_auc": round(float(roc_auc_score(y, s)), 6),
        "pr_auc": round(pr, 6),
        "lift": round(pr / base, 4),
    }


def fixture_eval(model_path: str, X, y, holdout: int, seed: int, testdata: str | None) -> dict:
    """Качество выгруженной модели на том же сплите, что и при обучении: train.py
    делит perm(seed)[:holdout] - повторяем и получаем метки holdout без файла."""
    import hashlib

    import lightgbm as lgb
    from sklearn.metrics import precision_recall_curve, roc_curve

    booster = lgb.Booster(model_file=model_path)
    hold = np.random.default_rng(seed).permutation(len(X))[: min(holdout, len(X) // 5)]
    X_ho, y_ho = X[hold], y[hold].astype(int)

    checks = {}
    if testdata:
        # Сверка с артефактами оригинального обучения: строки совпадают с
        # holdout.csv (пишется %.17g - точный round-trip float64; rtol лишь
        # страхует сравнение от смены формата), а предсказания - с ref_raw.csv.
        # Это и доказывает, что оценивается тот же сплит той же моделью.
        td = pathlib.Path(testdata)
        file_ho = np.loadtxt(td / "holdout.csv", delimiter=",", skiprows=1)
        if file_ho.shape != X_ho.shape:
            # testdata не от того обучения (например, фикстурный режим make data).
            checks["holdout_shape_mismatch"] = f"{file_ho.shape} vs {X_ho.shape}"
        else:
            recon = X_ho.astype("float64")
            same = np.isclose(recon, file_ho, rtol=1e-8, atol=1e-12) | (
                np.isnan(recon) & np.isnan(file_ho)
            )
            checks["holdout_rows_matched"] = int(same.all(axis=1).sum())
            checks["holdout_rows_total"] = int(len(file_ho))
            ref_raw = np.loadtxt(td / "ref_raw.csv", skiprows=1)
            checks["ref_raw_max_abs_diff"] = float(
                np.abs(booster.predict(X_ho, raw_score=True) - ref_raw).max()
            )

    s = booster.predict(X_ho, raw_score=True)
    out = eval_scores(y_ho, s)
    pred = s > 0  # дефолтный порог сервиса: margin > 0
    tp, fp = int((pred & (y_ho == 1)).sum()), int((pred & (y_ho == 0)).sum())
    fn = int((~pred & (y_ho == 1)).sum())
    out["margin0_precision"] = round(tp / (tp + fp), 4) if tp + fp else None
    out["margin0_recall"] = round(tp / (tp + fn), 4) if tp + fn else None
    out["margin0_declined"] = int(pred.sum())

    fpr, tpr, _ = roc_curve(y_ho, s)
    prec, rec, _ = precision_recall_curve(y_ho, s)
    out["roc_curve"] = dict(zip(("fpr", "tpr"), downsample(fpr, tpr)))
    out["pr_curve"] = dict(zip(("recall", "precision"), downsample(rec, prec)))

    gain = booster.feature_importance(importance_type="gain")
    share = gain / gain.sum()
    out["gain_share"] = {n: round(float(v), 4) for n, v in zip(booster.feature_name(), share)}
    out["model"] = model_path
    out["model_sha256_16"] = hashlib.sha256(pathlib.Path(model_path).read_bytes()).hexdigest()[:16]
    out["checks"] = checks
    return out


def split_comparison(X, y, ts, seed: int, threads: int, n_train: int, n_test: int) -> dict:
    """Случайный против временного сплита, одинаковое обучение - меняется только
    сплит. Временной: train на раннем периоде (до 85-го перцентиля времени),
    test на позднем - режим "предсказываем будущее" с дрейфом доли атак."""
    import lightgbm as lgb

    rng = np.random.default_rng(seed)
    params = {**TRAIN_PARAMS, "num_threads": threads, "seed": seed}

    def fit(idx):
        return lgb.train(params, lgb.Dataset(X[idx], label=y[idx]), num_boost_round=200)

    def sub(idx, n):
        return idx if len(idx) <= n else rng.choice(idx, n, replace=False)

    perm = rng.permutation(len(X))
    rnd_test, rnd_train = perm[:n_test], sub(perm[n_test:], n_train)

    order = np.argsort(ts, kind="stable")
    cutoff = ts[order[int(len(order) * 0.85)]]
    all_idx = np.arange(len(X))
    early, late = all_idx[ts < cutoff], all_idx[ts >= cutoff]
    tmp_train, tmp_test = sub(early, n_train), sub(late, n_test)

    m_rnd, m_tmp = fit(rnd_train), fit(tmp_train)
    return {
        "n_train_subsample": n_train,
        "n_test_subsample": n_test,
        "temporal_cutoff": str(cutoff),
        "attack_rate_early": round(float(y[early].mean()), 6),
        "attack_rate_late": round(float(y[late].mean()), 6),
        "random": eval_scores(y[rnd_test], m_rnd.predict(X[rnd_test], raw_score=True)),
        "temporal": eval_scores(y[tmp_test], m_tmp.predict(X[tmp_test], raw_score=True)),
        "leakage_inputs": {  # та же случайная пара train/test нужна контрасту утечки
            "rnd_train": rnd_train,
            "rnd_test": rnd_test,
        },
    }


def leakage_contrast(X, y, names, aux: dict, idx_train, idx_test, seed: int, threads: int) -> dict:
    """Возвращает выброшенные глобально-частотные признаки страны/ASN и меряет
    утечку: метка выводится из IP, страна и ASN - тоже, глобальная частота
    читает метку почти напрямую. Контраст к честной модели тех же параметров."""
    import lightgbm as lgb

    cols = []
    for key in ("country", "asn"):
        _, inv, cnt = np.unique(aux[key], return_inverse=True, return_counts=True)
        cols.append((cnt / len(aux[key]))[inv].astype("float32"))
    X14 = np.column_stack([X, *cols])
    names14 = [*names, "country_freq", "asn_freq"]

    params = {**TRAIN_PARAMS, "num_threads": threads, "seed": seed}
    m = lgb.train(
        params,
        lgb.Dataset(X14[idx_train], label=y[idx_train], feature_name=names14),
        num_boost_round=200,
    )
    out = eval_scores(y[idx_test], m.predict(X14[idx_test], raw_score=True))
    gain = m.feature_importance(importance_type="gain")
    share = gain / gain.sum()
    out["features_added"] = ["country_freq", "asn_freq"]
    out["freq_gain_share"] = round(float(share[-2] + share[-1]), 4)
    out["gain_share"] = {n: round(float(v), 4) for n, v in zip(names14, share)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="путь к rba-dataset.csv")
    ap.add_argument("--model", required=True, help="выгруженная модель (serving/fixtures/model.txt)")
    ap.add_argument("--outdir", required=True, help="куда писать validation.json (results)")
    ap.add_argument("--holdout", type=int, default=50000, help="как при обучении (make data-rba)")
    ap.add_argument("--seed", type=int, default=708)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--train-sub", type=int, default=3_000_000,
                    help="подвыборка обучения для сравнения сплитов и утечки")
    ap.add_argument("--test-sub", type=int, default=200_000)
    ap.add_argument("--testdata", help="каталог holdout.csv/ref_raw.csv оригинального обучения")
    ap.add_argument("--commit", help="git-коммит репозитория для провенанса")
    args = ap.parse_args()

    import lightgbm as lgb

    t0 = time.monotonic()
    # extras: коды страны/ASN, выровненные с X, - для контраста утечки ниже.
    X, y, ts, names, _codes, aux = load_rba(args.input, "attack", extras=True)
    t_load = time.monotonic() - t0

    t0 = time.monotonic()
    fx = fixture_eval(args.model, X, y, args.holdout, args.seed, args.testdata)
    t_fixture = time.monotonic() - t0

    t0 = time.monotonic()
    sc = split_comparison(X, y, ts, args.seed, args.threads, args.train_sub, args.test_sub)
    leak_in = sc.pop("leakage_inputs")
    t_split = time.monotonic() - t0

    t0 = time.monotonic()
    lk = leakage_contrast(X, y, names, aux, leak_in["rnd_train"], leak_in["rnd_test"],
                          args.seed, args.threads)
    t_leak = time.monotonic() - t0

    result = {
        "meta": {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "commit": args.commit,
            "lightgbm_version": lgb.__version__,
            "dataset": args.input,
            "dataset_rows": int(len(X)),
            "attack_rate": round(float(y.mean()), 6),
            "n_features": len(names),
            "feature_names": names,
            "seed": args.seed,
            "threads": args.threads,
            "durations_s": {
                "load": round(t_load, 1),
                "fixture_eval": round(t_fixture, 1),
                "split_comparison": round(t_split, 1),
                "leakage_contrast": round(t_leak, 1),
            },
        },
        "fixture_eval": fx,
        "split_comparison": sc,
        "leakage_contrast": lk,
    }
    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print(f"rows={len(X):,} attack_rate={y.mean():.4f} | load {t_load:.0f}s")
    print(f"fixture: ROC-AUC={fx['roc_auc']:.4f} PR-AUC={fx['pr_auc']:.4f} lift={fx['lift']:.2f} "
          f"checks={fx['checks']}")
    print(f"split:   random ROC={sc['random']['roc_auc']:.4f} lift={sc['random']['lift']:.2f} | "
          f"temporal ROC={sc['temporal']['roc_auc']:.4f} lift={sc['temporal']['lift']:.2f}")
    print(f"leakage: ROC={lk['roc_auc']:.4f} freq_gain_share={lk['freq_gain_share']:.3f}")
    print(f"wrote {out / 'validation.json'}")


if __name__ == "__main__":
    main()
