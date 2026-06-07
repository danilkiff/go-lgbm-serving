#!/usr/bin/env bash
# Скачивает датасет RBA (risk-based authentication, Wiefling et al.) в
# training/testdata/rba-dataset.csv.
# Нужен kaggle CLI и API-токен в ~/.kaggle/kaggle.json (chmod 600).
# См. https://www.kaggle.com/docs/api.
set -euo pipefail

dest="$(cd "$(dirname "$0")" && pwd)/testdata"
mkdir -p "$dest"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found." >&2
  echo "  install:  uv tool install kaggle   (or: pipx install kaggle)" >&2
  echo "  auth:     put your token at ~/.kaggle/kaggle.json (chmod 600)" >&2
  exit 1
fi

kaggle datasets download -d dasgroup/rba-dataset -p "$dest" --unzip
echo "wrote $dest/rba-dataset.csv"
