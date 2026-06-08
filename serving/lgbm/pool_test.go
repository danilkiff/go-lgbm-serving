package lgbm

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

// TestPoolConcurrentConsistency долбит пул Booster из множества горутин и
// проверяет, что каждое предсказание точно равно однопоточному эталону. Гонка на
// общем хэндле (отказ #3751) проявилась бы здесь несогласованными результатами
// для одного входа; схема пула "хэндл на горутину" этого не допускает.
//
// Запускать с -race, чтобы заодно проверить Go-код самого пула.
func TestPoolConcurrentConsistency(t *testing.T) {
	requireData(t)
	X := loadCSV(t, "holdout.csv")

	// Однопоточный эталон.
	ref, err := LoadBooster(tdPath("model.txt"))
	if err != nil {
		t.Fatalf("load ref: %v", err)
	}
	want := make([]float64, len(X))
	for i, row := range X {
		if want[i], err = ref.PredictRaw(row); err != nil {
			t.Fatalf("ref row %d: %v", i, err)
		}
	}
	ref.Close()

	pool, err := NewPool(tdPath("model.txt"), runtime.GOMAXPROCS(0))
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	defer pool.Close()

	const (
		workers = 64
		iters   = 500
	)
	var mismatches, calls int64
	var wg sync.WaitGroup
	for w := range workers {
		wg.Add(1)
		go func(seed int) {
			defer wg.Done()
			for it := range iters {
				i := (seed*7 + it*13) % len(X)
				got, err := pool.PredictRaw(X[i])
				atomic.AddInt64(&calls, 1)
				if err != nil || got != want[i] {
					atomic.AddInt64(&mismatches, 1)
				}
			}
		}(w)
	}
	wg.Wait()

	t.Logf("pool: %d workers x %d preds = %d calls across %d handles, %d mismatches",
		workers, iters, calls, runtime.GOMAXPROCS(0), mismatches)
	if mismatches != 0 {
		t.Fatalf("%d/%d predictions diverged under concurrency - shared predict state?", mismatches, calls)
	}
}
