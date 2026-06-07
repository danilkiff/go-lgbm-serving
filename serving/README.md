# Инференс RBA модели на Go

Что обеспечивает:

- инференс обученной в Python модели LightGBM через cgo (C-API);
- конвейер decline->explain и нативные SHAP-коды причин.

Здесь cgo линкует ту же `lib_lightgbm` из `training/.venv`, поэтому Go и Python исполняют один предиктор - отсюда битово точный паритет. Тесты паритета читают эталоны из `training/testdata` - сперва `make -C training data`.

## Команды

`make help` печатает список; основное:

```sh
make test    # паритет + юнит-тесты (-p 1)
make race    # детектор гонок на паритете и конкуренции
make bench   # задержка скоринга против SHAP, одиночный против пула
make run     # запуск на закоммиченной фикстуре, порт :8080
```

## REST API

Примеры запросов в `clients/http/scorer.http`.
