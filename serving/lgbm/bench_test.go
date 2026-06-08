package lgbm

import (
	"os"
	"runtime"
	"testing"
)

func benchData(b *testing.B) (*Booster, [][]float64) {
	b.Helper()
	if _, err := os.Stat(tdPath("model.txt")); err != nil {
		b.Skip("no testdata - run `make -C training data` first")
	}
	bo, err := LoadBooster(tdPath("model.txt"))
	if err != nil {
		b.Fatalf("load: %v", err)
	}
	X, err := readMatrix(tdPath("holdout.csv"))
	if err != nil {
		b.Fatalf("holdout: %v", err)
	}
	return bo, X
}

// BenchmarkPredictRaw и BenchmarkPredictContrib вместе отвечают на главный
// вопрос: насколько режим нативного SHAP (contrib) дороже обычного скоринга?
func BenchmarkPredictRaw(b *testing.B) {
	bo, X := benchData(b)
	defer bo.Close()
	b.ReportAllocs()
	for i := 0; b.Loop(); i++ {
		if _, err := bo.PredictRaw(X[i%len(X)]); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPredictContrib(b *testing.B) {
	bo, X := benchData(b)
	defer bo.Close()
	b.ReportAllocs()
	for i := 0; b.Loop(); i++ {
		if _, err := bo.PredictContrib(X[i%len(X)]); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkPoolRawParallel меряет пропускную способность, когда GOMAXPROCS
// горутин берут хэндлы из пула - форма боевого инференса.
func BenchmarkPoolRawParallel(b *testing.B) {
	if _, err := os.Stat(tdPath("model.txt")); err != nil {
		b.Skip("no testdata - run `make -C training data` first")
	}
	X, err := readMatrix(tdPath("holdout.csv"))
	if err != nil {
		b.Fatalf("holdout: %v", err)
	}
	pool, err := NewPool(tdPath("model.txt"), runtime.GOMAXPROCS(0))
	if err != nil {
		b.Fatalf("pool: %v", err)
	}
	defer pool.Close()

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			if _, err := pool.PredictRaw(X[i%len(X)]); err != nil {
				b.Fatal(err)
			}
			i++
		}
	})
}
