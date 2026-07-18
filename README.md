# go-lgbm-serving

[![CI](https://github.com/danilkiff/go-lgbm-serving/actions/workflows/ci.yml/badge.svg)](https://github.com/danilkiff/go-lgbm-serving/actions/workflows/ci.yml)

Инференс обученной в Python модели LightGBM из Go через cgo, с нативными
кодами причин SHAP и harness численного паритета.

Домен для модели - сессионный фрод /risk-based authentication (RBA).

## Проблема

> Может ли Go исполнять инференс модели, обученной в Python, воспроизводить
> её оценки и выдавать объяснимые коды причин на каждое решение - и с какими
> показателями быстродействия?

## Цели

- выбрать способ инференса на Go обученной в Python GBDT-модели и обеспечить
паритет веса contributions между Go и Python;
- доказать объяснимость через SHAP: на отклонение - код причины, доказуемо
связанный с решением (контракт объяснений - в [docs/DESIGN.md](docs/DESIGN.md),
"Результат");
- измерить быстродействие, найти точки деградации производительности и показать,
что архитектурные инварианты их закрывают.

Данные RBA синтетические, а качество обучения модели здесь вторично:
модели достаточно не быть абсурдной.

## TLDR: результаты

Модель обучена - качество скромное: ROC-AUC `0.703` на синтетическом RBA, ни один
признак не доминирует (max gain 33%). Цифры качества сведены с данными программно:
артефакт [validation.json](training/results/validation.json) и ноутбук
[analysis.ipynb](training/results/analysis.ipynb), воспроизводится
`make -C training validate notebook`.

Паритет битово точный на одной сборке (holdout 50000):

- raw margin maxD `0`;
- SHAP maxD `0`;
- sum(contrib)=margin maxD `1.5e-14`;
- смен решения `0`;
- топ-3 кодов `0/50000`.

Кросс-платформенно (darwin-arm64 против linux-amd64, та же `model.txt`, холдаут
4000 строк, 89% с NaN - missing-ветки деревьев под проверкой):

- raw margin совпадает до бита (maxD `0`, 0 смен);
- SHAP maxD `6.7e-16`;
- топ-3 идентичны (`make -C serving xparity-dump` на обеих платформах, сверка `xparity.py`).

Нативный SHAP на инференсе - только LightGBM/XGBoost (C-ABI `C_API_PREDICT_CONTRIB`),
CatBoost и ONNX не подходят.

Точка деградации: SHAP ~58x дороже скоринга (`PredictRaw` 16.3 мкс, `PredictContrib` 938 мкс).

Решения:

- пулинг: ~19x пропускной способности (0.86 мкс на предсказание, 32 потока), без
гонок (32k предсказаний под `-race`, 0 расхождений с эталоном);
- изоляция горячего пути закрывает стоимость SHAP: p99 `/score` под насыщенной
очередью explain ~20 мкс, неотличим от холостого, против ~930 мкс на один SHAP
(`TestHotPathIsolation`).

Провенанс перф-цифр: AMD Ryzen 9 5950X (16c/32t, Linux), LightGBM 4.6.0, Go 1.26 -
`make -C serving bench`. Абсолютные значения и множитель пулинга машинозависимы:
на Apple M4 Pro (12 потоков) - 6.3 мкс / 288 мкс (~45x), пул 0.94 мкс (~7x).

## См. также

Цель, метод, рассмотренные альтернативы и измеренный результат - в [docs/DESIGN.md](docs/DESIGN.md). Команды и устройство подсистем - в [serving/README.md](serving/README.md) и [training/README.md](training/README.md).

## Лицензия

[MIT](LICENSE).
