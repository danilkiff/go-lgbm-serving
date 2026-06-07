# Цели обучения: построение эталонных артефактов LightGBM в training/testdata/.
# Выполняются из каталога training/ (CWD = training/), поэтому `uv run` берёт
# здешний .venv. Из корня репозитория: `make -C training <цель>`.

.DEFAULT_GOAL := help
.PHONY: help data data-rba clean

help: ## показать цели этого Makefile
	@grep -hE '^[a-zA-Z][a-zA-Z_-]*:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

data: ## пересобрать эталоны на синтетике (для CI, без скачивания)
	uv run python train.py --dataset synthetic --outdir testdata

data-rba: ## скачать датасет RBA и обучить модель сессионного фрода (нужен kaggle CLI)
	./fetch_rba.sh
	uv run python train.py --dataset rba --input testdata/rba-dataset.csv --target attack --holdout 50000 --threads 8 --outdir testdata

clean: ## удалить сгенерированные артефакты (testdata/)
	rm -rf testdata
