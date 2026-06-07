# Флаги линковки cgo - путь к lib_lightgbm из uv-venv и rpath - заданы в директиве
# #cgo в serving/lgbm/lgbm.go, а НЕ здесь. Флаг в переменной CGO_LDFLAGS применяется go
# дважды (в cgo-объект и повторно на внешней линковке), из-за чего каждый -rpath
# дублировался и линковщик это предупреждал. Директива #cgo применяется один раз,
# поэтому сборка чистая и не требует настройки окружения: голый `go build ./...`
# работает сам по себе.

PLATFORM := $(shell go env GOOS)-$(shell go env GOARCH)

.PHONY: data data-rba test race bench bench-smoke run vet tidy clean print-env xparity-dump

data: ## пересобрать эталонные артефакты на синтетике (для CI, без скачивания)
	cd training && uv run python train.py --dataset synthetic

data-rba: ## скачать датасет RBA и обучить модель сессионного фрода (нужен kaggle CLI)
	./training/fetch_rba.sh
	cd training && uv run python train.py --dataset rba --input ../testdata/rba-dataset.csv --target attack --holdout 50000 --threads 8

# -p 1: пакеты не параллельно, иначе нативный SHAP в lgbm душит CPU и тайминг
# TestHotPathIsolation в pipeline флейкает.
test: ## прогнать паритет и юнит-тесты
	go -C serving test -v -p 1 ./...

race: ## прогнать паритет и тесты конкуренции под детектором гонок
	go -C serving test -race -run 'TestParity|TestPool|TestExplain|TestWorker' ./...

bench: ## бенчмарки (скоринг против contrib, одиночный против пула)
	go -C serving test -run=NONE -bench=. -benchmem ./...

bench-smoke: ## быстрый прогон бенчмарков для CI (компиляция и короткий прогон)
	go -C serving test -run=NONE -bench=. -benchtime=20x ./lgbm/

run: ## запустить сервис scorer: POST /score, GET /explain/{id} (доп. флаги через ARGS=)
	go -C serving run ./cmd/scorer $(ARGS)

xparity-dump: ## выгрузить предсказания ЭТОЙ платформы в testdata/pred.<os>-<arch>.csv
	go -C serving run ./cmd/dump ../testdata/model.txt ../testdata/holdout.csv ../testdata/pred.$(PLATFORM).csv

vet:
	go -C serving vet ./...

tidy:
	go -C serving mod tidy

clean:
	rm -rf testdata
	go -C serving clean ./...

print-env: ## проверить путь к нативной lib_lightgbm, с которой линкуется cgo (отладка сборки)
	@echo "cgo links lib_lightgbm via serving/lgbm/lgbm.go #cgo LDFLAGS (path is relative to the package dir)."
	@echo "uv-resolved lib dir (must match that path):"
	@cd training && uv run python -c "import lightgbm,pathlib;p=pathlib.Path(lightgbm.__file__).parent/'lib';print('  ', p, '(exists)' if p.exists() else '(MISSING)')"
