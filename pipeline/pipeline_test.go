package pipeline

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/danilkiff/go-lgbm-serving/lgbm"
)

func tdPool(t *testing.T) *lgbm.Pool { return tdPoolN(t, runtime.GOMAXPROCS(0)) }

func tdPoolN(t *testing.T, n int) *lgbm.Pool {
	t.Helper()
	model := filepath.Join("..", "testdata", "model.txt")
	if _, err := os.Stat(model); err != nil {
		t.Skip("no testdata - run `make data` first")
	}
	p, err := lgbm.NewPool(model, n)
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	return p
}

// TestScorerDeclineEmits форсирует отклонение (порог ниже любой маржи) и
// проверяет, что горячий путь выкладывает DeclineEvent с id решения, маржой,
// версией модели и копией строки.
func TestScorerDeclineEmits(t *testing.T) {
	pool := tdPool(t)
	defer pool.Close()
	row := make([]float64, pool.NumFeature())

	q := NewChannelQueue(8)
	s := NewScorer(pool, -1e18, "test-model", q)
	res, err := s.Score(row)
	if err != nil {
		t.Fatalf("score: %v", err)
	}
	if res.Decision != Decline {
		t.Fatalf("decision=%v, want decline", res.Decision)
	}
	select {
	case e := <-q.Events():
		if e.ID != res.ID {
			t.Errorf("event id %q != result id %q", e.ID, res.ID)
		}
		if e.Margin != res.Margin {
			t.Errorf("event margin %v != result margin %v", e.Margin, res.Margin)
		}
		if e.ModelVer != "test-model" {
			t.Errorf("event modelVer %q, want test-model", e.ModelVer)
		}
		if len(e.Row) != len(row) {
			t.Errorf("event row len %d, want %d", len(e.Row), len(row))
		}
	default:
		t.Fatal("decline did not emit an event")
	}
}

// TestScorerApproveNoEmit форсирует одобрение (порог выше любой маржи) и
// проверяет, что событие не выкладывается.
func TestScorerApproveNoEmit(t *testing.T) {
	pool := tdPool(t)
	defer pool.Close()
	row := make([]float64, pool.NumFeature())

	q := NewChannelQueue(8)
	s := NewScorer(pool, 1e18, "test-model", q)
	res, err := s.Score(row)
	if err != nil {
		t.Fatalf("score: %v", err)
	}
	if res.Decision != Approve {
		t.Fatalf("decision=%v, want approve", res.Decision)
	}
	select {
	case <-q.Events():
		t.Fatal("approve must not emit an event")
	default:
	}
}

// TestChannelQueueDropsWhenFull проверяет, что горячий путь защищён: полная
// очередь отбрасывает (не блокирует) и учитывает потерю.
func TestChannelQueueDropsWhenFull(t *testing.T) {
	q := NewChannelQueue(1)
	if !q.Publish(DeclineEvent{ID: "a"}) {
		t.Fatal("first publish should succeed")
	}
	if q.Publish(DeclineEvent{ID: "b"}) {
		t.Fatal("second publish should drop (buffer full)")
	}
	if q.Dropped() != 1 {
		t.Fatalf("dropped=%d, want 1", q.Dropped())
	}
}
