"""Инженерия признаков load_rba на крошечном синтетическом CSV.

Закрепляет причинность признаков: хронологический порядок на пользователя,
новизна - первое появление пары (user, категория), счётчики и паузы - только из
прошлого. Порядок строк в файле перемешан намеренно: сортировка обязана его
восстановить, иначе "новизна" подсматривала бы будущее.
"""

from __future__ import annotations

import numpy as np

from train import RBA_FEATURES, load_rba

HEADER = (
    "index,Login Timestamp,User ID,Round-Trip Time [ms],IP Address,Country,Region,City,"
    "ASN,User Agent String,Browser Name and Version,OS Name and Version,Device Type,"
    "Login Successful,Is Attack IP,Is Account Takeover"
)

# Хронологическая истина: user 1 - три входа (второй без RTT, третий из новой
# страны и с атакующего IP), user 2 - один вход из страны, уже виденной user 1
# (новизна обязана быть per-user). В файл строки пишутся перемешанными.
ROWS = {
    "u1t1": "0,2020-02-03 10:00:00.000,1,100.0,ip,NO,r,Oslo,100,ua,Chrome 79,Mac OS X,desktop,True,False,False",
    "u1t2": "1,2020-02-03 11:00:00.000,1,,ip,NO,r,Oslo,100,ua,Chrome 79,Mac OS X,desktop,False,False,True",
    "u1t3": "2,2020-02-04 11:00:00.000,1,50.0,ip,DE,r,Berlin,200,ua,Firefox 72,Windows 10,mobile,True,True,False",
    "u2t1": "3,2020-02-03 12:30:00.000,2,10.0,ip,NO,r,Bergen,100,ua,Firefox 72,Ubuntu,desktop,True,False,False",
}
FILE_ORDER = ["u1t3", "u2t1", "u1t1", "u1t2"]


def write_csv(tmp_path):
    p = tmp_path / "rba.csv"
    p.write_text("\n".join([HEADER, *(ROWS[k] for k in FILE_ORDER)]) + "\n")
    return str(p)


def test_features_causal_order(tmp_path):
    X, y, ts, names, codes = load_rba(write_csv(tmp_path), "attack")

    assert names == [f[0] for f in RBA_FEATURES]
    assert codes == {f[0]: {"code": f[1], "label": f[2]} for f in RBA_FEATURES}

    # Строки восстановлены в порядок (user, ts); столбцы - порядок RBA_FEATURES:
    # rtt, ok, hour, dow, n_prior, gap, new_country/city/asn/os/browser/device.
    nan = np.nan
    want = np.array(
        [
            [100.0, 1, 10, 0, 0, -1.0, 1, 1, 1, 1, 1, 1],  # u1t1: первый вход
            [nan, 0, 11, 0, 1, 3600.0, 0, 0, 0, 0, 0, 0],  # u1t2: всё знакомо, RTT не помечен
            [50.0, 1, 11, 1, 2, 86400.0, 1, 1, 1, 1, 1, 1],  # u1t3: всё новое для user 1
            [10.0, 1, 12, 0, 0, -1.0, 1, 1, 1, 1, 1, 1],  # u2t1: NO знакома user 1, но не user 2
        ],
        dtype="float32",
    )
    assert X.shape == want.shape
    assert np.array_equal(X, want, equal_nan=True), f"X=\n{X}"

    assert y.tolist() == [0, 0, 1, 0]
    assert (np.diff(ts[:3]).astype("timedelta64[s]").astype(int) > 0).all()


def test_target_ato(tmp_path):
    _, y, _, _, _ = load_rba(write_csv(tmp_path), "ato")
    assert y.tolist() == [0, 1, 0, 0]


def test_extras_aligned(tmp_path):
    X, _, _, _, _, aux = load_rba(write_csv(tmp_path), "attack", extras=True)

    assert len(aux["country"]) == len(X) and len(aux["asn"]) == len(X)
    # Словарные коды выровнены с восстановленным порядком строк: NO у обоих
    # пользователей кодируется одинаково, DE - иначе.
    c = aux["country"]
    assert c[0] == c[1] == c[3] and c[2] != c[0]
    assert aux["asn"].tolist() == [100, 100, 200, 100]
