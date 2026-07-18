package main

import (
	"bytes"
	"encoding/csv"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/danilkiff/go-lgbm-serving/lgbm"
)

func TestLoadCSV(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "x.csv")
	if err := os.WriteFile(p, []byte("a,b,c\n1,2,3\n4.5,-6,7e1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	rows, err := loadCSV(p)
	if err != nil {
		t.Fatalf("loadCSV: %v", err)
	}
	want := [][]float64{{1, 2, 3}, {4.5, -6, 70}}
	if len(rows) != len(want) {
		t.Fatalf("got %d rows, want %d", len(rows), len(want))
	}
	for i := range want {
		for j := range want[i] {
			if rows[i][j] != want[i][j] {
				t.Errorf("row %d col %d: got %v, want %v", i, j, rows[i][j], want[i][j])
			}
		}
	}
}

// TestWriteDump прогоняет сериализацию дампа на настоящей модели и проверяет
// ключевой инвариант: записанный raw margin равен сумме записанных SHAP contributions,
// построчно.
func TestWriteDump(t *testing.T) {
	model := filepath.Join("..", "..", "..", "training", "testdata", "model.txt")
	holdout := filepath.Join("..", "..", "..", "training", "testdata", "holdout.csv")
	if _, err := os.Stat(model); err != nil {
		t.Skip("no testdata - run `make -C training data` first")
	}
	b, err := lgbm.LoadBooster(model)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer b.Close()

	rows, err := loadCSV(holdout)
	if err != nil {
		t.Fatalf("holdout: %v", err)
	}
	if len(rows) > 100 {
		rows = rows[:100] // тест должен быть быстрым; SHAP в десятки раз дороже скоринга
	}

	var buf bytes.Buffer
	if err := writeDump(b, rows, &buf); err != nil {
		t.Fatalf("writeDump: %v", err)
	}

	recs, err := csv.NewReader(&buf).ReadAll()
	if err != nil {
		t.Fatalf("parse dump: %v", err)
	}
	wantCols := 1 + b.NumFeature() + 1 // raw + (признаки + база)
	if len(recs) != len(rows)+1 {
		t.Fatalf("got %d lines, want %d (+header)", len(recs), len(rows)+1)
	}
	if recs[0][0] != "raw" || len(recs[0]) != wantCols {
		t.Fatalf("header = %v (cols %d), want first col 'raw' and %d cols", recs[0], len(recs[0]), wantCols)
	}
	for i, rec := range recs[1:] {
		raw, _ := strconv.ParseFloat(rec[0], 64)
		var sum float64
		for _, s := range rec[1:] {
			v, _ := strconv.ParseFloat(s, 64)
			sum += v
		}
		if d := math.Abs(raw - sum); d > 1e-9 {
			t.Fatalf("row %d: raw %.6g != sum(contrib) %.6g (d %.2e)", i, raw, sum, d)
		}
	}
	t.Logf("dumped %d rows x %d cols; raw == sum(contrib) within 1e-9", len(rows), wantCols)
}
