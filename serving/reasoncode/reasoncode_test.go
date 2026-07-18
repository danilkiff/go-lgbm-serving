package reasoncode

import (
	"slices"
	"testing"
)

func TestTopK(t *testing.T) {
	// модули: [1, 5, 3, 5, 0]. По убыванию |v|, ничьи - меньший индекс:
	// idx1 и idx3 оба |5| -> 1 раньше 3; затем idx2 |3|.
	c := []float64{1, -5, 3, 5, 0}
	if got, want := TopK(c, 3), []int{1, 3, 2}; !slices.Equal(got, want) {
		t.Fatalf("TopK=%v want %v", got, want)
	}
}

// TestTopKPositive закрепляет доменный инвариант причин отклонения: только
// положительные contributions (толкавшие к decline), по убыванию значения,
// нулевые и отрицательные исключены, при нехватке результат короче k.
func TestTopKPositive(t *testing.T) {
	c := []float64{1, -5, 3, 5, 0}
	if got, want := TopKPositive(c, 3), []int{3, 2, 0}; !slices.Equal(got, want) {
		t.Fatalf("TopKPositive=%v want %v", got, want)
	}
	if got := TopKPositive(c, 10); len(got) != 3 {
		t.Fatalf("k>positives: got %v, want 3 indices", got)
	}
	if got := TopKPositive([]float64{-1, 0, -2}, 3); len(got) != 0 {
		t.Fatalf("no positives: got %v, want empty", got)
	}
	// Ничья по значению - меньший индекс первым.
	if got, want := TopKPositive([]float64{2, 2, 1}, 2), []int{0, 1}; !slices.Equal(got, want) {
		t.Fatalf("tie: got %v want %v", got, want)
	}
}

func TestTopKClampAndEmpty(t *testing.T) {
	c := []float64{2, 1}
	if got := TopK(c, 5); !slices.Equal(got, []int{0, 1}) {
		t.Fatalf("k>len: got %v want [0 1]", got)
	}
	if got := TopK(c, 0); len(got) != 0 {
		t.Fatalf("k=0: got %v want empty", got)
	}
	if got := TopK(nil, 3); len(got) != 0 {
		t.Fatalf("nil contrib: got %v want empty", got)
	}
}
