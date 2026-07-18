package lgbm

import (
	"encoding/csv"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"testing"

	"github.com/danilkiff/go-lgbm-serving/reasoncode"
)

// Допуск для паритета на одной сборке и платформе. На идентичном бинарнике
// liblightgbm это по сути шум округления float; между платформами отклонение
// будет больше (см. README, "TLDR: результаты") - в этом и смысл
// кросс-платформенной проверки, а не баг здесь.
const (
	rawTol     = 1e-6
	contribTol = 1e-6
	topK       = 3
)

type meta struct {
	LightGBMVersion  string `json:"lightgbm_version"`
	NFeatures        int    `json:"n_features"`
	NHoldout         int    `json:"n_holdout"`
	ContribShape     []int  `json:"contrib_shape"`
	ScoreIsRawMargin bool   `json:"score_is_raw_margin"`
}

func tdPath(name string) string { return filepath.Join("..", "..", "training", "testdata", name) }

func requireData(t *testing.T) {
	t.Helper()
	if _, err := os.Stat(tdPath("model.txt")); err != nil {
		t.Skip("no testdata - run `make -C training data` first")
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
	// Гейт семантики эталонов: сравниваем raw margin, а не вероятности.
	if !m.ScoreIsRawMargin {
		t.Fatal("meta.score_is_raw_margin=false, reference dumps must hold raw margins")
	}
	return m
}

// readMatrix читает числовой CSV с заголовком в строки float64. Без *testing.T -
// годится и для бенчмарков; loadCSV - обёртка для тестов.
func readMatrix(path string) ([][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	recs, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, err
	}
	if len(recs) < 1 {
		return nil, nil // пустой файл: нет заголовка - нет строк
	}
	out := make([][]float64, 0, len(recs)-1)
	for _, rec := range recs[1:] { // пропустить заголовок
		row := make([]float64, len(rec))
		for j, s := range rec {
			if row[j], err = strconv.ParseFloat(s, 64); err != nil {
				return nil, err
			}
		}
		out = append(out, row)
	}
	return out, nil
}

// loadCSV - обёртка readMatrix для тестов: проваливает тест при ошибке.
func loadCSV(t *testing.T, name string) [][]float64 {
	t.Helper()
	out, err := readMatrix(tdPath(name))
	if err != nil {
		t.Fatalf("load %s: %v", name, err)
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

	// Усечённый или пустой артефакт дал бы ноль сравнений и ложную зелень:
	// формы сверяются с meta до единого цикла.
	if len(X) == 0 || len(X) != m.NHoldout || len(ref) != len(X) {
		t.Fatalf("shape: holdout=%d ref_raw=%d meta.n_holdout=%d", len(X), len(ref), m.NHoldout)
	}

	var maxDiff, sumDiff float64
	flips := 0
	for i, row := range X {
		got, err := b.PredictRaw(row)
		if err != nil {
			t.Fatalf("row %d: %v", i, err)
		}
		want := ref[i][0]
		d := math.Abs(got - want)
		// NaN несравним: d>maxDiff всегда ложно, и нечисловое расхождение
		// проскочило бы молча.
		if math.IsNaN(d) {
			t.Fatalf("row %d: NaN in comparison (got=%v want=%v)", i, got, want)
		}
		sumDiff += d
		if d > maxDiff {
			maxDiff = d
		}
		// Решение - знак raw margin; flip - это изменившееся решение.
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
	// Формы против meta до цикла: усечённый артефакт не даст ложную зелень.
	if len(X) == 0 || len(X) != m.NHoldout || len(ref) != len(X) || len(raw) != len(X) {
		t.Fatalf("shape: holdout=%d ref_contrib=%d ref_raw=%d meta.n_holdout=%d",
			len(X), len(ref), len(raw), m.NHoldout)
	}
	if len(m.ContribShape) != 2 || m.ContribShape[0] != m.NHoldout || m.ContribShape[1] != width {
		t.Fatalf("meta.contrib_shape=%v, want [%d %d]", m.ContribShape, m.NHoldout, width)
	}

	var maxDiff, maxSumDiff float64
	topMismatch := 0
	for i, row := range X {
		got, err := b.PredictContrib(row)
		if err != nil {
			t.Fatalf("row %d: %v", i, err)
		}
		if len(got) != width || len(ref[i]) != width {
			t.Fatalf("row %d: contrib len %d, ref len %d, want %d", i, len(got), len(ref[i]), width)
		}
		// 1. поэлементный паритет с SHAP из Python
		var sum float64
		for j := range got {
			d := math.Abs(got[j] - ref[i][j])
			if math.IsNaN(d) {
				t.Fatalf("row %d col %d: NaN in comparison (got=%v want=%v)", i, j, got[j], ref[i][j])
			}
			if d > maxDiff {
				maxDiff = d
			}
			sum += got[j]
		}
		// 2. внутренний инвариант: sum(contrib) == raw margin
		d := math.Abs(sum - raw[i][0])
		if math.IsNaN(d) {
			t.Fatalf("row %d: NaN in sum(contrib) vs raw margin", i)
		}
		if d > maxSumDiff {
			maxSumDiff = d
		}
		// 3. устойчивость кодов причин: топ-K признаков по |contribution| (без base
		//    value) должны совпадать с эталоном Python.
		if !slices.Equal(
			reasoncode.TopK(got[:m.NFeatures], topK),
			reasoncode.TopK(ref[i][:m.NFeatures], topK),
		) {
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
