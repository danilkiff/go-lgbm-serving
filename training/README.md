# Обучение RBA модели на Python

Что делает:

- строит поведенческие RBA-признаки из сырого датасета;
- обучает LightGBM с фиксированным seed;
- выгружает эталонные артефакты, которые harness паритета на Go обязан воспроизвести. 

Среда - uv (`pyproject.toml`).

## Артефакты в `--outdir`

- `model.txt`;
- `holdout.csv`;
- `ref_raw.csv` (эталонная маржа);
- `ref_contrib.csv` (эталонный SHAP);
- `meta.json` (версия LightGBM, формы);
- `codes.json` (индекс признака -> код причины).

Обучение детерминировано: `deterministic=true`, `force_row_wise=true`, `num_threads` фиксирован.

## Команды

`make help` печатает список:

```sh
make data      # эталоны паритета из фикстуры -> testdata/ (для CI)
make data-rba  # скачать датасет RBA (~9 ГБ) и обучить (нужен kaggle CLI)
make clean     # удалить testdata/
```
