# Цели подачи: сборка, тесты и бенчи Go-модуля. Выполняются из serving/ (CWD =
# serving/). Из корня репозитория: `make -C serving <цель>`. Тесты читают эталоны
# из ../training/testdata - сначала `make -C training data`. cgo линкует
# lib_lightgbm из training/.venv (директива #cgo в lgbm/lgbm.go), флаги линковки
# тут НЕ задаются: голый `go build ./...` работает сам (CGO_LDFLAGS применялся бы
# дважды и дублировал -rpath).

PLATFORM := $(shell go env GOOS)-$(shell go env GOARCH)

.DEFAULT_GOAL := help
.PHONY: help test race bench bench-smoke run vet tidy clean print-env xparity-dump

help: ## показать цели этого Makefile
	@grep -hE '^[a-zA-Z][a-zA-Z_-]*:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# -p 1: пакеты не параллельно, иначе нативный SHAP в lgbm душит CPU и тайминг
# TestHotPathIsolation в pipeline флейкает.
test: ## паритет и юнит-тесты
	go test -v -p 1 ./...

race: ## паритет и тесты конкуренции под детектором гонок
	go test -race -run 'TestParity|TestPool|TestExplain|TestWorker' ./...

bench: ## бенчмарки (скоринг против contrib, одиночный против пула)
	go test -run=NONE -bench=. -benchmem ./...

bench-smoke: ## быстрый прогон бенчей для CI (компиляция + короткий прогон)
	go test -run=NONE -bench=. -benchtime=20x ./lgbm/

run: ## запустить scorer на фикстуре (флаги явные; доп. через ARGS=)
	go run ./cmd/scorer -model fixtures/model.txt -codes fixtures/codes.json $(ARGS)

vet: ## go vet
	go vet ./...

tidy: ## go mod tidy
	go mod tidy

clean: ## go clean
	go clean ./...

print-env: ## путь к нативной lib_lightgbm, с которой линкуется cgo (отладка сборки)
	@echo "cgo links lib_lightgbm via lgbm/lgbm.go #cgo LDFLAGS (path is relative to the package dir)."
	@echo "uv-resolved lib dir (must match that path):"
	@cd ../training && uv run python -c "import lightgbm,pathlib;p=pathlib.Path(lightgbm.__file__).parent/'lib';print('  ', p, '(exists)' if p.exists() else '(MISSING)')"

xparity-dump: ## дамп предсказаний этой платформы в ../training/testdata/pred.<os>-<arch>.csv
	go run ./cmd/dump ../training/testdata/model.txt ../training/testdata/holdout.csv ../training/testdata/pred.$(PLATFORM).csv
