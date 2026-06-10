"""Обучает LightGBM и выгружает артефакты для harness паритета на Go.

Эталонная сторона доказательства: Go грузит тот же model.txt через C API (cgo) и
обязан воспроизвести эти числа. Сверка - по raw margin (predict(raw_score=True) ==
C_API_PREDICT_RAW_SCORE), до сигмоиды, один в один; SHAP contributions
(predict(pred_contrib=True) == C_API_PREDICT_CONTRIB) суммируются в тот же margin - это
вторая, внутренняя проверка для стороны Go.

Режимы:
  --dataset rba|csv  обучить и выгрузить эталоны. rba: датасет RBA (Wiefling et al.),
                     цель Is Attack IP, признаки из сырых колонок (RTT, время суток,
                     новизна страны/города/ASN/ОС/браузера/устройства). csv: числовой
                     CSV с колонкой-целью.
  --from-model PATH  без обучения: выгрузить эталоны из готового model.txt на
                     случайном холдауте - так CI проверяет паритет на закоммиченной
                     фикстуре, не качая 9 ГБ датасета.

deterministic=True, force_row_wise=True, фиксированный num_threads - обучение
воспроизводимо при одном числе потоков. Битоточность между машинами требует той же
сборки liblightgbm; версию пишем в meta.json.

Выходы (в --outdir, обязателен): model.txt, holdout.csv (матрица признаков),
ref_raw.csv (эталонный raw margin), ref_contrib.csv (SHAP, форма n x nfeat+1, последний
столбец - base value), meta.json (версия/параметры/формы для сверки на Go),
codes.json (индекс -> код причины, только при обучении).
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def load_csv(path: str, target: str):
    """Адаптер реального открытого датасета. Только числовые столбцы."""
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


# Инженерные признаки RBA в порядке столбцов holdout.csv и индексов кодов причин.
# Код причины - стабильный идентификатор adverse-action; метка - человекочитаемая.
RBA_FEATURES = [
    ("rtt_ms", "RTT", "round-trip time входа, мс"),
    ("login_successful", "LOGINOK", "вход успешен"),
    ("hour", "HOUR", "час суток входа"),
    ("dow", "DOW", "день недели входа"),
    ("n_prior_logins", "NPRIOR", "число прошлых входов пользователя"),
    ("secs_since_last", "GAP", "секунд с прошлого входа пользователя"),
    ("is_new_country", "NEWCTRY", "новая страна для пользователя"),
    ("is_new_city", "NEWCITY", "новый город для пользователя"),
    ("is_new_asn", "NEWASN", "новый ASN для пользователя"),
    ("is_new_os", "NEWOS", "новая ОС для пользователя"),
    ("is_new_browser", "NEWBR", "новый браузер для пользователя"),
    ("is_new_device", "NEWDEV", "новый тип устройства для пользователя"),
]


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _resolve(header, want):
    """Сопоставляет логические имена столбцов фактическим по нормализованной подстроке.
    Терпимо к точной пунктуации заголовка (например, 'Round-Trip Time [ms]')."""
    norm = {_norm(h): h for h in header}
    out = {}
    for key, sub in want.items():
        match = next((orig for n, orig in norm.items() if sub in n), None)
        if match is None:
            raise SystemExit(f"RBA: не найден столбец для {key!r} (подстрока {sub!r}) в {header}")
        out[key] = match
    return out


def load_rba(path: str, target: str):
    """Строит числовые RBA-признаки из сырого датасета RBA. Цель: 'attack'
    (Is Attack IP) или 'ato' (Is Account Takeover).

    Признаки причинны по времени: новизна и счётчики считаются в хронологическом
    порядке на пользователя, поэтому используют только прошлое - без утечки метки.
    Сырые IP и полная строка User-Agent выброшены: высокая кардинальность,
    избыточность с ОС/браузером и близость к метке (она задаётся по IP)."""
    import pyarrow.csv as pacsv
    import pyarrow as pa
    import pandas as pd

    label_col = "ato" if target == "ato" else "attack"
    want = {
        "user": "userid",
        "ts": "timestamp",
        "rtt": "roundtrip",
        "country": "country",
        "city": "city",
        "asn": "asn",
        "os": "osname",
        "browser": "browsername",
        "device": "devicetype",
        "ok": "loginsuccessful",
        "attack": "isattackip",
        "ato": "isaccounttakeover",
    }
    header = pd.read_csv(path, nrows=0).columns.tolist()
    col = _resolve(header, want)
    include = [col[k] for k in ("user", "ts", "rtt", "country", "city", "asn",
                                "os", "browser", "device", "ok", label_col)]

    tbl = pacsv.read_csv(
        path,
        convert_options=pacsv.ConvertOptions(include_columns=include, strings_can_be_null=True),
    )

    def num(name):
        return tbl[name].to_numpy(zero_copy_only=False)

    def codes(name):
        # Строку -> целочисленный код через словарное кодирование Arrow, без
        # материализации 33M python-строк. null -> -1.
        d = tbl[name].combine_chunks().dictionary_encode()
        idx = d.indices.to_numpy(zero_copy_only=False).astype("float64")
        return np.nan_to_num(idx, nan=-1).astype("int32")

    def boolean(name):
        c = tbl[name]
        if pa.types.is_boolean(c.type):
            return c.to_numpy(zero_copy_only=False).astype("int8")
        d = c.combine_chunks().dictionary_encode()
        vals = [str(v).lower() in ("true", "1") for v in d.dictionary.to_pylist()]
        lut = np.array(vals + [False], dtype="int8")  # последний - для null-индекса -1
        idx = d.indices.to_numpy(zero_copy_only=False).astype("float64")
        return lut[np.nan_to_num(idx, nan=len(vals)).astype("int64")]

    # Login Timestamp - строка datetime ("2020-02-03 12:43:30.772"), не epoch.
    tcol = tbl[col["ts"]]
    ts = tcol.to_pandas() if pa.types.is_timestamp(tcol.type) else \
        pd.to_datetime(tcol.to_pandas(), errors="coerce")

    d = pd.DataFrame({
        "user": num(col["user"]).astype("int64"),
        "ts": ts.to_numpy(),
        "rtt": num(col["rtt"]).astype("float32"),
        "ok": boolean(col["ok"]),
        "y": boolean(col[label_col]),
        "country": codes(col["country"]),
        "city": codes(col["city"]),
        "asn": np.nan_to_num(num(col["asn"]).astype("float64"), nan=-1).astype("int64"),
        "os": codes(col["os"]),
        "browser": codes(col["browser"]),
        "device": codes(col["device"]),
    })
    del tbl

    # Хронологический порядок на пользователя - основа причинных признаков.
    d.sort_values(["user", "ts"], kind="stable", inplace=True)
    g = d.groupby("user", sort=False)
    gap = g["ts"].diff().dt.total_seconds()  # секунд с прошлого входа; NaN у первого

    feat = pd.DataFrame(index=d.index)
    feat["rtt_ms"] = d["rtt"]
    feat["login_successful"] = d["ok"]
    feat["hour"] = d["ts"].dt.hour.fillna(-1).astype("int8")
    feat["dow"] = d["ts"].dt.dayofweek.fillna(-1).astype("int8")
    feat["n_prior_logins"] = g.cumcount().astype("int32")
    feat["secs_since_last"] = gap.fillna(-1.0).astype("float32")
    for name, cat in (("is_new_country", "country"), ("is_new_city", "city"),
                      ("is_new_asn", "asn"), ("is_new_os", "os"),
                      ("is_new_browser", "browser"), ("is_new_device", "device")):
        # Первое хронологическое появление (user, категория) = новизна (использует
        # только прошлое: при сортировке по времени "первое" не подсматривает будущее).
        feat[name] = (~d.duplicated(["user", cat], keep="first")).astype("int8")

    names = [f[0] for f in RBA_FEATURES]
    X = feat[names].to_numpy(dtype="float32")
    y = d["y"].to_numpy().astype("int32")
    ts = d["ts"].to_numpy()  # datetime64, выровнен с X - для временного сплита и дрейфа
    reason_codes = {f[0]: {"code": f[1], "label": f[2]} for f in RBA_FEATURES}
    return X, y, ts, names, reason_codes


def _sample_holdout(rng, n: int, names):
    """Случайные строки в форме признаков модели - правдоподобные диапазоны на имя
    признака RBA, для незнакомого - стандартная нормаль. Для паритета (Go == Python)
    важна только форма входа, не реализм значений."""
    binary = {"login_successful", "is_new_country", "is_new_city", "is_new_asn",
              "is_new_os", "is_new_browser", "is_new_device"}
    cols = []
    for nm in names:
        if nm in binary:
            c = rng.integers(0, 2, n)
        elif nm == "hour":
            c = rng.integers(0, 24, n)
        elif nm == "dow":
            c = rng.integers(0, 7, n)
        elif nm == "n_prior_logins":
            c = rng.integers(0, 500, n)
        elif nm == "secs_since_last":
            c = rng.exponential(3600.0, n)
            c[rng.random(n) < 0.1] = -1.0  # первый вход пользователя
        elif nm == "rtt_ms":
            # В реальном RBA RTT ~96% null: без NaN паритет не заходил бы в
            # missing-ветки деревьев (default_left) - а это основной маршрут.
            c = rng.gamma(2.0, 50.0, n)
            c[rng.random(n) < 0.9] = np.nan
        else:
            c = rng.standard_normal(n)
        cols.append(np.asarray(c, dtype="float64"))
    return np.column_stack(cols)


def dump_from_model(model_path: str, outdir: str, n_rows: int, seed: int) -> None:
    """Выгружает эталоны паритета из готовой модели, без обучения. Холдаут -
    случайные строки в форме признаков модели; эталоны считаются на той же сборке
    liblightgbm, что грузит Go-тест, поэтому паритет битоточный, а датасет RBA для
    этой проверки не нужен."""
    import shutil

    import lightgbm as lgb

    booster = lgb.Booster(model_file=model_path)
    names = booster.feature_name()
    X = _sample_holdout(np.random.default_rng(seed), n_rows, names)
    raw = booster.predict(X, raw_score=True)
    contrib = booster.predict(X, pred_contrib=True)

    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_path, out / "model.txt")  # тот же байт-в-байт файл, что грузит Go
    # %.17g: полный round-trip float64, чтобы Go читал те же входы -> паритет битоточный.
    np.savetxt(out / "holdout.csv", X, delimiter=",", header=",".join(names), comments="", fmt="%.17g")
    np.savetxt(out / "ref_raw.csv", raw, delimiter=",", header="raw_margin", comments="", fmt="%.17g")
    np.savetxt(out / "ref_contrib.csv", contrib, delimiter=",",
               header=",".join([*names, "base_value"]), comments="", fmt="%.17g")
    meta = {
        "lightgbm_version": lgb.__version__,
        "source": f"from-model:{model_path}",
        "seed": seed,
        "n_features": len(names),
        "feature_names": names,
        "n_holdout": int(n_rows),
        "score_is_raw_margin": True,
        "contrib_shape": list(contrib.shape),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"lightgbm {lgb.__version__} | from-model {model_path} | {len(names)} features, {n_rows} holdout rows")
    print(f"wrote {out}/ : model.txt, holdout.csv, ref_raw.csv, ref_contrib.csv {contrib.shape}, meta.json")
    print(f"raw margin range [{raw.min():.4f}, {raw.max():.4f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-model",
                    help="выгрузить эталоны паритета из готового model.txt, без обучения")
    ap.add_argument("--dataset", choices=["rba", "csv"], default="rba")
    ap.add_argument("--input", help="CSV path for --dataset rba|csv")
    ap.add_argument("--target", default="attack",
                    help="rba: 'attack' (Is Attack IP) or 'ato'; csv: target column name")
    ap.add_argument("--holdout", type=int, default=4000,
                    help="rows held out for the parity check (или сгенерированных при --from-model)")
    ap.add_argument("--seed", type=int, default=708)
    ap.add_argument("--threads", type=int, default=1, help="LightGBM num_threads (fixed for reproducibility)")
    ap.add_argument("--outdir", required=True,
                    help="output dir for reference artifacts (e.g. testdata, relative to CWD)")
    args = ap.parse_args()

    if args.from_model:
        dump_from_model(args.from_model, args.outdir, args.holdout, args.seed)
        return

    import lightgbm as lgb

    reason_codes = None
    if args.dataset == "rba":
        if not args.input:
            raise SystemExit("--dataset rba requires --input <path to rba-dataset.csv>")
        X, y, _ts, names, reason_codes = load_rba(args.input, args.target)
    else:  # csv
        if not args.input:
            raise SystemExit("--dataset csv requires --input <path>")
        X, y, names = load_csv(args.input, args.target)
    if reason_codes is None:
        reason_codes = {n: {"code": f"R{i}", "label": n} for i, n in enumerate(names)}

    # Детерминированное разбиение holdout.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    n_hold = min(args.holdout, len(X) // 5)
    hold_idx, train_idx = perm[:n_hold], perm[n_hold:]
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_ho, y_ho = X[hold_idx], y[hold_idx]

    params = {
        "objective": "binary",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_data_in_leaf": 50,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,  # bagging многопоточный и невоспроизводимый - выключаем
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": args.threads,
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

    # Качество на holdout - для понимания модели, не часть проверки паритета.
    metrics = {"holdout_pos": int(y_ho.sum())}
    if 0 < y_ho.sum() < len(y_ho):
        from sklearn.metrics import average_precision_score, roc_auc_score

        pred = raw > 0  # порог решения сервиса по умолчанию: margin > 0
        tp = int((pred & (y_ho == 1)).sum())
        fp = int((pred & (y_ho == 0)).sum())
        fn = int((~pred & (y_ho == 1)).sum())
        metrics.update(
            roc_auc=float(roc_auc_score(y_ho, raw)),
            pr_auc=float(average_precision_score(y_ho, raw)),
            precision_at0=(tp / (tp + fp)) if tp + fp else None,
            recall_at0=(tp / (tp + fn)) if tp + fn else None,
        )

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
    codes = {str(i): reason_codes[n] for i, n in enumerate(names)}
    (out / "codes.json").write_text(json.dumps(codes, ensure_ascii=False, indent=2))
    meta = {
        "lightgbm_version": lgb.__version__,
        "dataset": args.dataset,
        "target": args.target,
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
        "holdout_metrics": metrics,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"lightgbm {lgb.__version__} | dataset={args.dataset} target={meta['target']}")
    print(f"wrote {out}/ : model.txt, holdout.csv ({X_ho.shape}), ref_raw.csv, ref_contrib.csv {contrib.shape}, meta.json, codes.json")
    print(f"train rows={len(train_idx)} pos_rate={y_tr.mean():.5f} | raw margin range [{raw.min():.4f}, {raw.max():.4f}]")
    print(f"holdout metrics: {metrics}")


if __name__ == "__main__":
    main()
