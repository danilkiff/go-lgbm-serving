#!/usr/bin/env bash
# Скачивает датасет RBA (risk-based authentication, Wiefling et al.) в
# training/testdata/rba-dataset.csv. Если файл уже есть - скачивание пропускается.
# Нужен kaggle CLI с авторизацией: `kaggle auth login` (либо ~/.kaggle/access_token /
# env KAGGLE_API_TOKEN; legacy ~/.kaggle/kaggle.json тоже работает).
# См. https://www.kaggle.com/docs/api.
set -euo pipefail

dest="$(cd "$(dirname "$0")" && pwd)/testdata"
mkdir -p "$dest"

if [ -f "$dest/rba-dataset.csv" ]; then
  echo "rba-dataset.csv уже есть в $dest - скачивание пропущено (удали файл, чтобы перекачать)"
  exit 0
fi

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found." >&2
  echo "  install:  uv tool install kaggle   (or: pipx install kaggle)" >&2
  echo "  auth:     kaggle auth login   (либо ~/.kaggle/access_token)" >&2
  exit 1
fi

kaggle datasets download -d dasgroup/rba-dataset -p "$dest" --unzip
echo "wrote $dest/rba-dataset.csv"
