package pipeline

import (
	"errors"
	"math"
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
// версией модели и копией строки: событие живёт дольше вызова Score, а
// вызывающий переиспользует срез - алиас дал бы объяснение чужого входа.
func TestScorerDeclineEmits(t *testing.T) {
	pool := tdPool(t)
	defer pool.Close()
	row := make([]float64, pool.NumFeature())
	for i := range row {
		row[i] = float64(i)
	}

	q := NewChannelQueue(8)
	s := NewScorer(pool, -1e18, "test-model", q)
	res, err := s.Score(row)
	if err != nil {
		t.Fatalf("score: %v", err)
	}
	if res.Decision != Decline {
		t.Fatalf("decision=%v, want decline", res.Decision)
	}
	row[0] = -42 // вызывающий переиспользовал срез после Score
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
			t.Fatalf("event row len %d, want %d", len(e.Row), len(row))
		}
		for i := range e.Row {
			if e.Row[i] != float64(i) {
				t.Errorf("event row[%d]=%v, want %v (copy, not alias)", i, e.Row[i], float64(i))
			}
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

// TestScorerThresholdBoundary закрепляет строгость порога: margin, равный
// threshold, одобряется; decline только при margin строго выше (контракт флага
// -threshold: "decline if margin > threshold"). Margin детерминирован на одном
// хэндле, поэтому равенство воспроизводимо точно.
func TestScorerThresholdBoundary(t *testing.T) {
	pool := tdPoolN(t, 1)
	defer pool.Close()
	row := make([]float64, pool.NumFeature())
	margin, err := pool.PredictRaw(row)
	if err != nil {
		t.Fatalf("margin: %v", err)
	}

	res, err := NewScorer(pool, margin, "m", nil).Score(row)
	if err != nil {
		t.Fatalf("score: %v", err)
	}
	if res.Decision != Approve {
		t.Fatalf("margin == threshold: decision=%v, want approve", res.Decision)
	}

	below := math.Nextafter(margin, math.Inf(-1))
	if res, err = NewScorer(pool, below, "m", nil).Score(row); err != nil {
		t.Fatalf("score: %v", err)
	}
	if res.Decision != Decline {
		t.Fatalf("threshold один ulp ниже margin: decision=%v, want decline", res.Decision)
	}
}

// TestScorerFeatureCountError: неверная ширина входа доходит до вызывающего как
// lgbm.ErrFeatureCount сквозь реальную цепочку пул -> Scorer. На этом различении
// HTTP-слой строит 422 против 500; фейк в cmd/scorer лишь повторяет обёртку -
// настоящую проверяет этот тест.
func TestScorerFeatureCountError(t *testing.T) {
	pool := tdPoolN(t, 1)
	defer pool.Close()
	s := NewScorer(pool, 0, "m", nil)
	if _, err := s.Score([]float64{1}); !errors.Is(err, lgbm.ErrFeatureCount) {
		t.Fatalf("err=%v, want errors.Is(ErrFeatureCount)", err)
	}
}

// TestChannelQueueDropsWhenFull проверяет, что горячий путь защищён: полная
// очередь отбрасывает (не блокирует), учитывает потерю и отдаёт отброшенное
// событие в OnDrop - потеря объяснения видна по id, а не только счётчиком.
func TestChannelQueueDropsWhenFull(t *testing.T) {
	q := NewChannelQueue(1)
	var droppedID string
	q.OnDrop = func(e DeclineEvent) { droppedID = e.ID }
	if !q.Publish(DeclineEvent{ID: "a"}) {
		t.Fatal("first publish should succeed")
	}
	if q.Publish(DeclineEvent{ID: "b"}) {
		t.Fatal("second publish should drop (buffer full)")
	}
	if q.Dropped() != 1 {
		t.Fatalf("dropped=%d, want 1", q.Dropped())
	}
	if droppedID != "b" {
		t.Fatalf("OnDrop got id=%q, want b", droppedID)
	}
}
