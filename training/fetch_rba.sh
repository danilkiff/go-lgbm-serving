#!/usr/bin/env bash
# Скачивает датасет RBA (risk-based authentication, Wiefling et al.) в
# testdata/rba-dataset.csv.
#
# Kaggle: dasgroup/rba-dataset - синтезированные данные входов: >33M попыток,
# 3.3M пользователей, Норвегия SSO, фев 2020 - фев 2021. CSV ~9 ГБ. Метки фрода:
# Is Attack IP (атакующий IP) и Is Account Takeover (захват аккаунта). Значения
# синтетические, не для боевых IDS. Лицензия CC BY 4.0; в репозитории НЕ хранятся.
#
# Нужен kaggle CLI и API-токен в ~/.kaggle/kaggle.json (chmod 600).
# См. https://www.kaggle.com/docs/api.
set -euo pipefail

dest="$(cd "$(dirname "$0")/.." && pwd)/testdata"
mkdir -p "$dest"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found." >&2
  echo "  install:  uv tool install kaggle   (or: pipx install kaggle)" >&2
  echo "  auth:     put your token at ~/.kaggle/kaggle.json (chmod 600)" >&2
  exit 1
fi

kaggle datasets download -d dasgroup/rba-dataset -p "$dest" --unzip
echo "wrote $dest/rba-dataset.csv"
