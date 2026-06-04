#!/usr/bin/env bash
# Скачивает датасет ULB Credit Card Fraud в testdata/creditcard.csv.
#
# Kaggle: mlg-ulb/creditcardfraud - 284807 транзакций, 30 числовых признаков
# (Time, V1..V28 PCA, Amount), цель "Class" (~0.17% положительных).
# Распространяется по Database Contents License; в этом репозитории НЕ хранится.
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

kaggle datasets download -d mlg-ulb/creditcardfraud -p "$dest" --unzip
echo "wrote $dest/creditcard.csv"
