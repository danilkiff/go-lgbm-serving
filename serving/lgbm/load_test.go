package lgbm

import (
	"os"
	"strings"
	"testing"
)

// TestLoadBoosterFromBytesMatchesFile: загрузка из байтов эквивалентна загрузке
// из файла - предсказание совпадает до бита. Через этот путь идут пул и версия
// модели (одни байты для хеша и хэндлов).
func TestLoadBoosterFromBytesMatchesFile(t *testing.T) {
	const path = "../fixtures/model.txt"
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	bf, err := LoadBooster(path)
	if err != nil {
		t.Fatalf("from file: %v", err)
	}
	defer bf.Close()
	bb, err := LoadBoosterFromBytes(data)
	if err != nil {
		t.Fatalf("from bytes: %v", err)
	}
	defer bb.Close()

	row := make([]float64, bf.NumFeature())
	want, err := bf.PredictRaw(row)
	if err != nil {
		t.Fatalf("file predict: %v", err)
	}
	got, err := bb.PredictRaw(row)
	if err != nil {
		t.Fatalf("bytes predict: %v", err)
	}
	if got != want {
		t.Fatalf("bytes booster %v != file booster %v", got, want)
	}
}

// TestLoadBoosterRejectsMulticlass: модель с несколькими выходами на строку
// отвергается на загрузке, а не молча теряет классы в проде. Фикстура -
// настоящая трёхклассовая модель (training/multiclass_fixture.py).
func TestLoadBoosterRejectsMulticlass(t *testing.T) {
	_, err := LoadBooster("../fixtures/multiclass.txt")
	if err == nil {
		t.Fatal("multiclass model must fail to load")
	}
	// Именно отказ по числу выходов: сбой чтения файла тоже дал бы err != nil.
	if !strings.Contains(err.Error(), "expected 1") {
		t.Fatalf("err=%v, want the multi-output rejection", err)
	}
}
