.DEFAULT_GOAL := help
.PHONY: help data data-rba validate notebook clean

help: ## показать цели этого Makefile
	@grep -hE '^[a-zA-Z][a-zA-Z_-]*:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

data: ## выгрузить эталоны паритета из закоммиченной RBA-фикстуры (для CI)
	uv run python train.py --from-model ../serving/fixtures/model.txt --outdir testdata

data-rba: ## скачать датасет RBA и обучить модель сессионного фрода (нужен kaggle CLI)
	./fetch_rba.sh
	uv run python train.py --dataset rba --input testdata/rba-dataset.csv --target attack --holdout 50000 --threads 8 --outdir testdata

validate: ## валидация качества на полном датасете -> results/validation.json (сперва data-rba)
	uv run python validate.py --input testdata/rba-dataset.csv \
		--model ../serving/fixtures/model.txt --testdata testdata \
		--outdir results --commit $$(git rev-parse --short HEAD)

notebook: ## пересобрать results/analysis.ipynb из analysis.py и исполнить с выводами
	uv run --group analysis jupytext --to ipynb -o results/analysis.ipynb results/analysis.py
	uv run --group analysis jupyter nbconvert --to notebook --execute --inplace results/analysis.ipynb

clean: ## удалить сгенерированные артефакты (testdata/)
	rm -rf testdata
