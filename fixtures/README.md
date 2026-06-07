# Фикстуры для примеров

`model.txt` + `codes.json` - выгруженная RBA-модель (12 поведенческих признаков,
цель Is Attack IP), закоммиченная, чтобы примеры `clients/http/` и `clients/postman/` работали на
свежем клоне без обучения и без скачивания датасета (~9 ГБ через `make data-rba`).

Это та же модель, что `make data-rba` кладёт в `testdata/model.txt`: детерминирована
(seed 708, 200 раундов) и проходит harness паритета. Лежит здесь, а не в
`testdata/`, потому что `make data` (синтетика для CI) перезаписывает `testdata/`
30-фичевой моделью, против которой 12-фичевые примеры не работают.

## Запуск scorer на фикстуре

```sh
go -C serving run ./cmd/scorer -model ../fixtures/model.txt -codes ../fixtures/codes.json
```

Затем гонять `clients/http/scorer.http` (JetBrains HTTP Client) или
`clients/postman/` (порт по умолчанию :8080).

## Обновить (после переобучения RBA)

```sh
make data-rba
cp testdata/model.txt testdata/codes.json fixtures/
```

`codes.json` - индекс признака -> код причины (`NPRIOR`, `GAP`, ...), порождается
`training/train.py` из списка `RBA_FEATURES`.
