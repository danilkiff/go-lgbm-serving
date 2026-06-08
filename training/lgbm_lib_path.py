"""Печатает каталог нативной lib_lightgbm в uv-окружении и помечает, существует
ли он. Путь обязан совпадать с тем, по которому cgo линкует lib_lightgbm
(директива #cgo LDFLAGS в serving/lgbm/lgbm.go) - отладка сборки.

    uv run python lgbm_lib_path.py
"""

from __future__ import annotations

import pathlib

import lightgbm


def main() -> None:
    p = pathlib.Path(lightgbm.__file__).parent / "lib"
    print("  ", p, "(exists)" if p.exists() else "(MISSING)")


if __name__ == "__main__":
    main()
