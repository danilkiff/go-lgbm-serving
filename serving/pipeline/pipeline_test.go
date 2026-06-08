package pipeline

import (
	"path/filepath"
	"runtime"
	"testing"

	"github.com/danilkiff/go-lgbm-serving/lgbm"
)

func tdPool(t *testing.T) *lgbm.Pool { return tdPoolN(t, runtime.GOMAXPROCS(0)) }

// tdPoolN строит пул на закоммиченной фикстуре (serving/fixtures/model.txt). Тесты
// конвейера проверяют механику (решение, очередь, воркеры), а не паритет, поэтому им
// годится любая рабочая модель - фикстура избавляет их от шага `make -C training data`.
func tdPoolN(t *testing.T, n int) *lgbm.Pool {
	t.Helper()
	model := filepath.Join("..", "fixtures", "model.txt")
	p, err := lgbm.NewPool(model, n)
	if err != nil {
		t.Fatalf("load fixture %s: %v", model, err)
	}
	return p
}

// TestScorerDeclineEmits форсирует отклонение (порог ниже любого margin) и
// проверяет, что горячий путь выкладывает DeclineEvent с id решения, margin,
// версией модели и копией строки.
func TestScorerDeclineEmits(t *testing.T) {
	pool := tdPool(t)
	defer pool.Close()
	row := make([]float64, pool.NumFeature())

	q := NewChannelQueue(8)
	s := NewScorer(pool, -1e18, "test-model", q, nil)
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

// TestScorerApproveNoEmit форсирует одобрение (порог выше любого margin) и
// проверяет, что событие не выкладывается.
func TestScorerApproveNoEmit(t *testing.T) {
	pool := tdPool(t)
	defer pool.Close()
	row := make([]float64, pool.NumFeature())

	q := NewChannelQueue(8)
	s := NewScorer(pool, 1e18, "test-model", q, nil)
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

// TestScorerOnDropWhenQueueFull проверяет: когда очередь полна и событие
// отброшено, Score уведомляет onDrop тем же id, что вернул клиенту, - потеря
// объяснения видна по id, а не только в агрегате queue_dropped.
func TestScorerOnDropWhenQueueFull(t *testing.T) {
	pool := tdPool(t)
	defer pool.Close()
	row := make([]float64, pool.NumFeature())

	q := NewChannelQueue(1) // буфер на одно событие, потребителя нет
	var dropped []DeclineEvent
	s := NewScorer(pool, -1e18, "m", q, func(e DeclineEvent) { dropped = append(dropped, e) })

	// Первое отклонение занимает буфер; событие никто не вычёрпывает.
	if _, err := s.Score(row); err != nil {
		t.Fatalf("score 1: %v", err)
	}
	// Второе отклонение не влезает -> отброшено -> onDrop с его id.
	second, err := s.Score(row)
	if err != nil {
		t.Fatalf("score 2: %v", err)
	}

	if len(dropped) != 1 {
		t.Fatalf("onDrop called %d times, want 1 (first decline fit the buffer)", len(dropped))
	}
	if dropped[0].ID != second.ID {
		t.Fatalf("onDrop id=%q, want dropped decline id=%q", dropped[0].ID, second.ID)
	}
	if q.Dropped() != 1 {
		t.Fatalf("queue dropped=%d, want 1", q.Dropped())
	}
}
