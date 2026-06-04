# Флаги линковки cgo - путь к lib_lightgbm из uv-venv и rpath - заданы в директиве
# #cgo в lgbm/lgbm.go, а НЕ здесь. Флаг в переменной CGO_LDFLAGS применяется go
# дважды (в cgo-объект и повторно на внешней линковке), из-за чего каждый -rpath
# дублировался и линковщик это предупреждал. Директива #cgo применяется один раз,
# поэтому сборка чистая и не требует настройки окружения: голый `go build ./...`
# работает сам по себе.

PLATFORM := $(shell go env GOOS)-$(shell go env GOARCH)

.PHONY: data data-creditcard test race bench bench-smoke run vet tidy clean print-env xparity-dump

data: ## пересобрать эталонные артефакты из python/train.py (синтетика)
	cd python && uv run python train.py

data-creditcard: ## скачать датасет ULB creditcard и обучиться на нём (нужен kaggle CLI)
	./python/fetch_creditcard.sh
	cd python && uv run python train.py --dataset csv --input ../testdata/creditcard.csv --target Class

test: ## прогнать паритет и юнит-тесты
	go test -v ./...

race: ## прогнать паритет и тесты конкуренции под детектором гонок
	go test -race -run 'TestParity|TestPool|TestExplain|TestWorker' ./...

bench: ## бенчмарки (скоринг против contrib, одиночный против пула)
	go test -run=NONE -bench=. -benchmem ./...

bench-smoke: ## быстрый прогон бенчмарков для CI (компиляция и короткий прогон)
	go test -run=NONE -bench=. -benchtime=20x ./lgbm/

run: ## запустить сервис scorer: POST /score, GET /explain/{id} (доп. флаги через ARGS=)
	go run ./cmd/scorer $(ARGS)

xparity-dump: ## выгрузить предсказания ЭТОЙ платформы в testdata/pred.<os>-<arch>.csv
	go run ./cmd/dump testdata/model.txt testdata/holdout.csv testdata/pred.$(PLATFORM).csv

vet:
	go vet ./...

tidy:
	go mod tidy

clean:
	rm -rf testdata
	go clean ./...

print-env: ## проверить путь к нативной lib_lightgbm, с которой линкуется cgo (отладка сборки)
	@echo "cgo links lib_lightgbm via lgbm/lgbm.go #cgo LDFLAGS (path is relative to the package dir)."
	@echo "uv-resolved lib dir (must match that path):"
	@cd python && uv run python -c "import lightgbm,pathlib;p=pathlib.Path(lightgbm.__file__).parent/'lib';print('  ', p, '(exists)' if p.exists() else '(MISSING)')"
