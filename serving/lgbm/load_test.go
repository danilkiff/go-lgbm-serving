package lgbm

import (
	"strings"
	"testing"
)

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
