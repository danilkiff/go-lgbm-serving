# Фикстуры для примеров

> После повторного обучения модели (папка training) фикстуры есть смысл обновить.

Здесь:

- `model.txt` - выгруженная RBA-модель (12 поведенческих признаков,
цель Is Attack IP). Нужна, чтобы примеры `serving/clients/http/` работали
сразу без обучения и без скачивания датасета (~9 ГБ через `make -C training data-rba`);
- `codes.json` - индекс признака -> код причины (`NPRIOR`, `GAP`, ...), порождается
`training/train.py` из списка `RBA_FEATURES`;
- `multiclass.txt` - трёхклассовая модель для негативного теста загрузки
(`TestLoadBoosterRejectsMulticlass`: несколько выходов на строку - отказ),
порождается `training/multiclass_fixture.py`.
