# Обучение RBA модели на Python

Что делает:

- строит поведенческие RBA-признаки из сырого датасета;
- обучает LightGBM с фиксированным seed;
- выгружает эталонные артефакты, которые harness паритета на Go обязан воспроизвести. 

Среда - uv (`pyproject.toml`).

## Артефакты в `--outdir`

- `model.txt`;
- `holdout.csv`;
- `ref_raw.csv` (эталонный raw margin);
- `ref_contrib.csv` (эталонный SHAP);
- `meta.json` (версия LightGBM, формы);
- `codes.json` (индекс признака -> код причины).

Обучение детерминировано: `deterministic=true`, `force_row_wise=true`, `num_threads` фиксирован.

## Валидация качества

`validate.py` сводит цифры качества README/DESIGN с данными: оценка фикстурной
модели на том же сплите, что при обучении (сверка с `holdout.csv`/`ref_raw.csv`),
случайный против временного сплита, контраст утечки частотных признаков.
Закоммиченные артефакты: `results/validation.json` (сырые числа) и
`results/analysis.ipynb` (рендер из JSON; исходник - `results/analysis.py`).

## Команды

`make help` печатает список:

```sh
make data      # эталоны паритета из фикстуры -> testdata/ (для CI)
make data-rba  # скачать датасет RBA (~9 ГБ) и обучить (нужен kaggle CLI)
make validate  # полный датасет (сперва data-rba) -> results/validation.json
make notebook  # пересобрать и исполнить results/analysis.ipynb из analysis.py
make clean     # удалить testdata/
```
