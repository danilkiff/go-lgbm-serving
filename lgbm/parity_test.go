package lgbm

import (
	"encoding/csv"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/danilkiff/go-lgbm-serving/reasoncode"
)

// Допуск для паритета на одной сборке и платформе. На идентичном бинарнике
// liblightgbm это по сути шум округления float; между платформами отклонение
// будет больше (см. README, "Численный паритет") - в этом и смысл
// кросс-платформенной проверки, а не баг здесь.
const (
	rawTol     = 1e-6
	contribTol = 1e-6
	topK       = 3
)

type meta struct {
	LightGBMVersion  string   `json:"lightgbm_version"`
	NFeatures        int      `json:"n_features"`
	FeatureNames     []string `json:"feature_names"`
	NHoldout         int      `json:"n_holdout"`
	ContribShape     []int    `json:"contrib_shape"`
	ScoreIsRawMargin bool     `json:"score_is_raw_margin"`
}

func tdPath(name string) string { return filepath.Join("..", "testdata", name) }

func requireData(t *testing.T) {
	t.Helper()
	if _, err := os.Stat(tdPath("model.txt")); err != nil {
		t.Skip("no testdata - run `make data` (python/train.py) first")
	}
}

func loadMeta(t *testing.T) meta {
	t.Helper()
	b, err := os.ReadFile(tdPath("meta.json"))
	if err != nil {
		t.Fatalf("read meta: %v", err)
	}
	var m meta
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("parse meta: %v", err)
	}
	return m
}

// loadCSV читает числовой CSV с заголовком в строки float64.
func loadCSV(t *testing.T, name string) [][]float64 {
	t.Helper()
	f, err := os.Open(tdPath(name))
	if err != nil {
		t.Fatalf("open %s: %v", name, err)
	}
	defer f.Close()
	r := csv.NewReader(f)
	recs, err := r.ReadAll()
	if err != nil {
		t.Fatalf("read %s: %v", name, err)
	}
	out := make([][]float64, 0, len(recs)-1)
	for _, rec := range recs[1:] { // пропустить заголовок
		row := make([]float64, len(rec))
		for j, s := range rec {
			v, err := strconv.ParseFloat(s, 64)
			if err != nil {
				t.Fatalf("%s: parse %q: %v", name, s, err)
			}
			row[j] = v
		}
		out = append(out, row)
	}
	return out
}

func TestParityRaw(t *testing.T) {
	requireData(t)
	m := loadMeta(t)
	b, err := LoadBooster(tdPath("model.txt"))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer b.Close()

	if b.NumFeature() != m.NFeatures {
		t.Fatalf("feature count: Go %d vs meta %d", b.NumFeature(), m.NFeatures)
	}

	X := loadCSV(t, "holdout.csv")
	ref := loadCSV(t, "ref_raw.csv") // n x 1

	var maxDiff, sumDiff float64
	flips := 0
	for i, row := range X {
		got, err := b.PredictRaw(row)
		if err != nil {
			t.Fatalf("row %d: %v", i, err)
		}
		want := ref[i][0]
		d := math.Abs(got - want)
		sumDiff += d
		if d > maxDiff {
			maxDiff = d
		}
		// Решение - знак сырой маржи; flip - это изменившееся решение.
		if (got > 0) != (want > 0) {
			flips++
		}
	}
	n := len(X)
	t.Logf("lightgbm %s | rows=%d | raw margin: maxD=%.3e meanD=%.3e | decision flips=%d",
		m.LightGBMVersion, n, maxDiff, sumDiff/float64(n), flips)
	if maxDiff > rawTol {
		t.Fatalf("raw parity: maxD %.3e > tol %.0e", maxDiff, rawTol)
	}
	if flips != 0 {
		t.Fatalf("decision flips: %d (must be 0 at same-build parity)", flips)
	}
}

func TestParityContrib(t *testing.T) {
	requireData(t)
	m := loadMeta(t)
	b, err := LoadBooster(tdPath("model.txt"))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer b.Close()

	X := loadCSV(t, "holdout.csv")
	ref := loadCSV(t, "ref_contrib.csv") // n x (nfeat+1)
	raw := loadCSV(t, "ref_raw.csv")

	width := m.NFeatures + 1
	var maxDiff, maxSumDiff float64
	topMismatch := 0
	for i, row := range X {
		got, err := b.PredictContrib(row)
		if err != nil {
			t.Fatalf("row %d: %v", i, err)
		}
		if len(got) != width {
			t.Fatalf("row %d: contrib len %d, want %d", i, len(got), width)
		}
		// 1. поэлементный паритет с SHAP из Python
		var sum float64
		for j := range got {
			if d := math.Abs(got[j] - ref[i][j]); d > maxDiff {
				maxDiff = d
			}
			sum += got[j]
		}
		// 2. внутренний инвариант: sum(contrib) == сырая маржа
		if d := math.Abs(sum - raw[i][0]); d > maxSumDiff {
			maxSumDiff = d
		}
		// 3. устойчивость кодов причин: топ-K признаков по |вкладу| (без базового
		//    члена) должны совпадать с эталоном Python.
		if !equalInts(reasoncode.TopK(got[:m.NFeatures], topK), reasoncode.TopK(ref[i][:m.NFeatures], topK)) {
			topMismatch++
		}
	}
	t.Logf("contrib: maxD=%.3e | sum(contrib)=raw invariant maxD=%.3e | top-%d reason-code mismatches=%d/%d",
		maxDiff, maxSumDiff, topK, topMismatch, len(X))
	if maxDiff > contribTol {
		t.Fatalf("contrib parity: maxD %.3e > tol %.0e", maxDiff, contribTol)
	}
	if maxSumDiff > 1e-5 {
		t.Fatalf("sum(contrib)!=raw margin: maxD %.3e", maxSumDiff)
	}
	if topMismatch != 0 {
		t.Fatalf("reason-code ordering mismatches: %d (must be 0 at same-build parity)", topMismatch)
	}
}

// equalInts сообщает, равны ли два среза индексов поэлементно.
func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
