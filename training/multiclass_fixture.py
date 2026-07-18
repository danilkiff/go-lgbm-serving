"""Обучает крошечную мультиклассовую модель - фикстуру негативного теста загрузки.

LoadBooster на стороне Go отказывается грузить модель с несколькими выходами на
строку: PredictRaw молча отдавал бы только первый класс. Проверить отказ можно
лишь настоящим мультиклассовым model.txt - его и порождает этот скрипт
(serving/fixtures/multiclass.txt, тест lgbm.TestLoadBoosterRejectsMulticlass).

    uv run python multiclass_fixture.py ../serving/fixtures/multiclass.txt
"""

from __future__ import annotations

import sys

import numpy as np


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: multiclass_fixture.py <out model.txt>")

    import lightgbm as lgb

    rng = np.random.default_rng(708)
    X = rng.standard_normal((300, 4))
    y = rng.integers(0, 3, 300)
    booster = lgb.train(
        {
            "objective": "multiclass",
            "num_class": 3,
            "num_leaves": 4,
            "min_data_in_leaf": 20,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": 1,
            "seed": 708,
            "verbose": -1,
        },
        lgb.Dataset(X, label=y),
        num_boost_round=2,
    )
    booster.save_model(sys.argv[1])
    print(f"wrote {sys.argv[1]}: 3 classes x 2 rounds")


if __name__ == "__main__":
    main()
