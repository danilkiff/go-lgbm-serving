package reasoncode

import (
	"reflect"
	"testing"
)

func TestTopK(t *testing.T) {
	// модули: [1, 5, 3, 5, 0]. По убыванию |v|, ничьи - меньший индекс:
	// idx1 и idx3 оба |5| -> 1 раньше 3; затем idx2 |3|.
	c := []float64{1, -5, 3, 5, 0}
	if got, want := TopK(c, 3), []int{1, 3, 2}; !reflect.DeepEqual(got, want) {
		t.Fatalf("TopK=%v want %v", got, want)
	}
}

func TestTopKClampAndEmpty(t *testing.T) {
	c := []float64{2, 1}
	if got := TopK(c, 5); !reflect.DeepEqual(got, []int{0, 1}) {
		t.Fatalf("k>len: got %v want [0 1]", got)
	}
	if got := TopK(c, 0); len(got) != 0 {
		t.Fatalf("k=0: got %v want empty", got)
	}
	if got := TopK(nil, 3); len(got) != 0 {
		t.Fatalf("nil contrib: got %v want empty", got)
	}
}
