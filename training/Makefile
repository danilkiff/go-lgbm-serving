.DEFAULT_GOAL := help
.PHONY: help data data-rba clean

help: ## показать цели этого Makefile
	@grep -hE '^[a-zA-Z][a-zA-Z_-]*:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

data: ## выгрузить эталоны паритета из закоммиченной RBA-фикстуры (для CI)
	uv run python train.py --from-model ../serving/fixtures/model.txt --outdir testdata

data-rba: ## скачать датасет RBA и обучить модель сессионного фрода (нужен kaggle CLI)
	./fetch_rba.sh
	uv run python train.py --dataset rba --input testdata/rba-dataset.csv --target attack --holdout 50000 --threads 8 --outdir testdata

clean: ## удалить сгенерированные артефакты (testdata/)
	rm -rf testdata
