.DEFAULT_GOAL := help
.PHONY: help data bench matrix notebook clean

help: ## показать цели этого Makefile
	@grep -hE '^[a-zA-Z][a-zA-Z_-]*:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

data: ## сгенерировать вход k6 из training/testdata -> testdata/ (rows.json + threshold)
	./scripts/gen-rows.sh

# Один прогон = матрица 2x2 (mem/pg x без лимита/1cpu+1gb) для текущих EXPLAIN/RATE.
# Параметры - через env перед make: EXPLAIN=inline RATE=4000 make bench.
bench: ## один прогон матрицы (env EXPLAIN=async|inline RATE=<rps>; по умолч. async/1000)
	./scripts/bench.sh

matrix: ## полная матрица: {async,inline} x {1000,4000} rps
	for e in async inline; do for r in 1000 4000; do EXPLAIN=$$e RATE=$$r ./scripts/bench.sh; done; done

notebook: ## открыть разбор результатов (jupyter; окружение из uv.lock)
	uv run jupyter notebook results/analysis.ipynb

clean: ## удалить сгенерированный вход (testdata/)
	rm -rf testdata
