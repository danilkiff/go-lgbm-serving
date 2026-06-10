#!/usr/bin/env bash
# Bootstrap всего пайплайна на чистой машине:
#   датасет RBA (~9 ГБ) -> обучение модели -> эталоны паритета -> тесты на Go.
#
# Требует: uv, go >= 1.26 (как в serving/go.mod), kaggle CLI с авторизацией
# (варианты - в fetch_rba.sh),
# рантайм OpenMP (Linux: libgomp; macOS: brew install libomp).
# Повторный запуск идемпотентен: уже скачанный датасет fetch_rba.sh пропускает.
set -euo pipefail
cd "$(dirname "$0")"

say() { printf '\n==> %s\n' "$*"; }

say "проверка инструментов"
miss=0
for c in uv go; do command -v "$c" >/dev/null 2>&1 || { echo "  нет в PATH: $c" >&2; miss=1; }; done
[ "$miss" = 0 ] || { echo "установи недостающее (или поправь PATH) и повтори" >&2; exit 1; }
go version
uv --version

say "1/4 датасет RBA + обучение модели  (make -C training data-rba)"
make -C training data-rba

say "2/4 паритет + юнит-тесты            (make -C serving test)"
make -C serving test

say "3/4 детектор гонок                  (make -C serving race)"
make -C serving race

say "4/4 бенчмарки                       (make -C serving bench)"
make -C serving bench

say "готово: датасет скачан, модель обучена, паритет/гонки/бенчи зелёные"
