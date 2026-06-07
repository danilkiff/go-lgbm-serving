# Ноутбуки

Парные jupytext: источник - `.py` (percent-формат), рядом сгенерированный `.ipynb`
с выводами. Правим `.py` - `.ipynb` пересобирается. Заголовок `.py` фиксирует пару
`py:percent,ipynb` и kernel `lgbm-rba`.

- `rba_validation` - методика оценки: случайный против временного сплита, кривые
  ROC/PR, калибровка, базлайны, помесячный дрейф, scale_pos_weight, SHAP beeswarm.
- `rba_quality` - качество выгруженной `model.txt` на holdout: сводные метрики,
  рабочие точки и подбор порога под целевую метрику, lift/gain, разделимость
  скоров, срезы, bootstrap-CI, анализ ошибок, per-decision SHAP на фродовых
  сессиях.

Оба ноутбука тяжёлые: грузят весь датасет RBA (~31M строк) через `load_rba` из
`training/train.py`. Нужны артефакты в `testdata/` (сначала `make data-rba`), для
`rba_quality` - ещё `model.txt` / `holdout.csv` / `ref_*.csv`. Рассчитаны на машину
с большой памятью.

Окружение - uv-venv в `training/` (там `pyproject.toml`). Все команды ниже - из
каталога `training/`:

```sh
cd training
```

## Генерация ноутбука из .py

```sh
uv run jupytext --to notebook notebooks/rba_quality.py   # .py -> .ipynb (без выводов)
uv run jupytext --sync     notebooks/rba_quality.py      # двусторонняя синхронизация .py <-> .ipynb
```

## Запуск ноутбука из CLI (с выводами)

Один раз зарегистрировать kernel, объявленный в метаданных ноутбука:

```sh
uv run python -m ipykernel install --user --name lgbm-rba --display-name lgbm-rba
```

Исполнить и вписать выводы (текст + графики) в `.ipynb` на месте:

```sh
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=lgbm-rba \
  --ExecutePreprocessor.timeout=3600 \
  notebooks/rba_quality.ipynb
```

Либо одной командой из `.py` (конвертация + исполнение сразу):

```sh
uv run jupytext --to notebook --execute notebooks/rba_quality.py
```

Интерактивно (правка в браузере; при сохранении jupytext синхронит `.ipynb` и `.py`):

```sh
uv run jupyter lab   # открыть notebooks/*.ipynb
```

Локально вместо `uv run <cmd>` можно звать бинарь venv напрямую: `./.venv/bin/<cmd>`.
