# Фикстуры для примеров

`model.txt` + `codes.json` - выгруженная RBA-модель (12 поведенческих признаков,
цель Is Attack IP), закоммиченная, чтобы примеры `serving/clients/http/` работали
на свежем клоне без обучения и без скачивания датасета (~9 ГБ через
`make -C training data-rba`).

Это та же модель, что `make -C training data-rba` кладёт в `training/testdata/model.txt`:
детерминирована (seed 708, 200 раундов) и проходит harness паритета. Лежит здесь, а
не в `training/testdata/`, потому что `make -C training data` (синтетика для CI)
перезаписывает `training/testdata/` 30-фичевой моделью, против которой 12-фичевые
примеры не работают.

## Запуск scorer на фикстуре

Флаги явные (`-model` обязателен); `-codes` опционален - без него коды причин
обобщённые (R<idx>):

```sh
go -C serving run ./cmd/scorer -model fixtures/model.txt -codes fixtures/codes.json
```

То же короче - `make -C serving run` (цель подставляет те же явные флаги).

Затем гонять `serving/clients/http/scorer.http` (JetBrains HTTP Client; порт по
умолчанию :8080).

## Обновить (после переобучения RBA)

```sh
make -C training data-rba
cp training/testdata/model.txt training/testdata/codes.json serving/fixtures/
```

`codes.json` - индекс признака -> код причины (`NPRIOR`, `GAP`, ...), порождается
`training/train.py` из списка `RBA_FEATURES`.
