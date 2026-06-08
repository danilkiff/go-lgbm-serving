package pipeline

import (
	"context"
	"fmt"
	"math"
	"sort"
	"testing"
	"time"

	"github.com/danilkiff/go-lgbm-serving/lgbm"
	"github.com/danilkiff/go-lgbm-serving/reasoncode"
)

func TestMemStore(t *testing.T) {
	s := NewMemStore()
	if _, ok := s.Get("x"); ok {
		t.Fatal("empty store returned a value")
	}
	s.Put(Explanation{ID: "x", Margin: 1.5})
	if got, ok := s.Get("x"); !ok || got.Margin != 1.5 {
		t.Fatalf("get=%+v ok=%v", got, ok)
	}
}

// TestExplainEndToEnd собирает весь путь decline->explain: отклонение на горячем
// пути выкладывает событие, воркер считает нативный SHAP вне пути и сохраняет
// объяснение, а мы проверяем сквозные инварианты - sum(contrib) примерно равно
// margin решения, и сохранённые коды причин равны топ-K заново посчитанных
// contributions. (TestParityContrib уже доказывает, что эти топ-K совпадают с эталоном
// Python, так что сохранённые коды совпадают с ним транзитивно.)
func TestExplainEndToEnd(t *testing.T) {
	pool := tdPool(t)
	row := make([]float64, pool.NumFeature())

	queue := NewChannelQueue(8)
	store := NewMemStore()
	const k = 3
	scorer := NewScorer(pool, -1e18, "test-model", queue) // порог ниже любого margin -> decline
	worker := NewWorker(pool, store, WorkerConfig{K: k})  // nil-каталог -> обобщённые коды

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { worker.Run(ctx, queue.Events()); close(done) }()
	// Остановить воркер до закрытия пула на всех путях выхода.
	defer func() { cancel(); <-done; pool.Close() }()

	res, err := scorer.Score(row)
	if err != nil {
		t.Fatalf("score: %v", err)
	}
	if res.Decision != Decline {
		t.Fatalf("decision=%v, want decline", res.Decision)
	}

	var exp Explanation
	var ok bool
	for i := 0; i < 200; i++ { // согласованность в конечном счёте: опрос до ~1с
		if exp, ok = store.Get(res.ID); ok {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if !ok {
		t.Fatal("explanation never appeared in store")
	}

	// Независимо пересчитать contributions и проверить инварианты.
	contrib, err := pool.PredictContrib(row)
	if err != nil {
		t.Fatalf("contrib: %v", err)
	}
	nf := len(contrib) - 1
	var sum float64
	for _, c := range contrib {
		sum += c
	}
	if d := math.Abs(sum - exp.Margin); d > 1e-5 {
		t.Errorf("sum(contrib)=%.6g vs served margin=%.6g (d=%.2e)", sum, exp.Margin, d)
	}
	if exp.Base != contrib[nf] {
		t.Errorf("base=%v, want contrib[-1]=%v", exp.Base, contrib[nf])
	}
	want := reasoncode.TopK(contrib[:nf], k)
	if len(exp.Reasons) != len(want) {
		t.Fatalf("reasons len=%d, want %d", len(exp.Reasons), len(want))
	}
	for i := range want {
		if exp.Reasons[i].Feature != want[i] {
			t.Errorf("reason %d: feature=%d, want %d", i, exp.Reasons[i].Feature, want[i])
		}
		if exp.Reasons[i].Contribution != contrib[want[i]] {
			t.Errorf("reason %d: contribution=%v, want %v", i, exp.Reasons[i].Contribution, contrib[want[i]])
		}
		if wantCode := fmt.Sprintf("R%d", want[i]); exp.Reasons[i].Code != wantCode {
			t.Errorf("reason %d: code=%q, want %q (generic, nil catalog)", i, exp.Reasons[i].Code, wantCode)
		}
		if wantDir := reasoncode.Direction(contrib[want[i]]); exp.Reasons[i].Direction != wantDir {
			t.Errorf("reason %d: direction=%q, want %q", i, exp.Reasons[i].Direction, wantDir)
		}
	}
}

// TestWorkerDeadLetters форсирует детерминированный сбой PredictContrib (строка
// неверной ширины) и проверяет, что событие повторяется, уходит в dead-letter,
// учитывается и никогда не сохраняется - код причины отклонения не теряется молча.
func TestWorkerDeadLetters(t *testing.T) {
	pool := tdPool(t)
	defer pool.Close()
	store := NewMemStore()

	var got DeclineEvent
	var gotErr error
	calls := 0
	w := NewWorker(pool, store, WorkerConfig{
		K:          3,
		Retries:    2,
		DeadLetter: func(e DeclineEvent, err error) { got, gotErr, calls = e, err, calls+1 },
	})

	bad := DeclineEvent{ID: "bad", Row: []float64{1, 2, 3}, Margin: 1, ModelVer: "m"}
	w.process(bad)

	if calls != 1 {
		t.Fatalf("dead-letter called %d times, want 1", calls)
	}
	if got.ID != "bad" || gotErr == nil {
		t.Fatalf("dead-letter event=%+v err=%v", got, gotErr)
	}
	if w.Dropped() != 1 {
		t.Fatalf("dropped=%d, want 1", w.Dropped())
	}
	if _, ok := store.Get("bad"); ok {
		t.Fatal("a failed event must not be stored")
	}
}

// TestWorkerPoolProcessesAll гоняет пул воркеров на пачке отклонений и проверяет,
// что каждое объяснено.
func TestWorkerPoolProcessesAll(t *testing.T) {
	pool := tdPool(t)
	row := make([]float64, pool.NumFeature())
	store := NewMemStore()
	queue := NewChannelQueue(64)
	w := NewWorker(pool, store, WorkerConfig{K: 3})

	ctx, cancel := context.WithCancel(context.Background())
	wait := w.Start(ctx, queue.Events(), 4)
	defer func() { cancel(); wait(); pool.Close() }()

	const m = 20
	for i := 0; i < m; i++ {
		queue.Publish(DeclineEvent{ID: fmt.Sprintf("e%d", i), Row: row, Margin: -1})
	}

	stored := func() int {
		n := 0
		for i := 0; i < m; i++ {
			if _, ok := store.Get(fmt.Sprintf("e%d", i)); ok {
				n++
			}
		}
		return n
	}
	for i := 0; i < 200 && stored() < m; i++ {
		time.Sleep(5 * time.Millisecond)
	}
	if n := stored(); n != m {
		t.Fatalf("explained %d/%d events", n, m)
	}
}

func scoreP99(t *testing.T, s *Scorer, row []float64, n int) time.Duration {
	t.Helper()
	d := make([]time.Duration, n)
	for i := range d {
		start := time.Now()
		if _, err := s.Score(row); err != nil {
			t.Fatalf("score: %v", err)
		}
		d[i] = time.Since(start)
	}
	sort.Slice(d, func(i, j int) bool { return d[i] < d[j] })
	return d[int(float64(n)*0.99)]
}

func medianContrib(t *testing.T, p *lgbm.Pool, row []float64, n int) time.Duration {
	t.Helper()
	d := make([]time.Duration, n)
	for i := range d {
		start := time.Now()
		if _, err := p.PredictContrib(row); err != nil {
			t.Fatalf("contrib: %v", err)
		}
		d[i] = time.Since(start)
	}
	sort.Slice(d, func(i, j int) bool { return d[i] < d[j] })
	return d[n/2]
}

// TestHotPathIsolation - ключевое свойство: насыщенная очередь explain (нативный
// SHAP примерно в 58 раз дороже скоринга) не должна попадать на путь /score.
// Меряет p99 /score вхолостую против полной нагрузки explain и проверяет, что он
// остаётся сильно ниже стоимости одного SHAP - то есть SHAP не на горячем пути.
func TestHotPathIsolation(t *testing.T) {
	if testing.Short() {
		t.Skip("timing test")
	}
	pool := tdPoolN(t, 8) // хэндлов больше, чем воркеров explain, чтобы горячий путь не голодал
	row := make([]float64, pool.NumFeature())

	const n = 2000
	base := scoreP99(t, NewScorer(pool, 1e18, "m", nil), row, n) // одобряем всё, без нагрузки explain

	queue := NewChannelQueue(1024)
	store := NewMemStore()
	scorer := NewScorer(pool, -1e18, "m", queue) // отклоняем всё -> насыщаем объяснитель
	w := NewWorker(pool, store, WorkerConfig{K: 3})
	ctx, cancel := context.WithCancel(context.Background())
	wait := w.Start(ctx, queue.Events(), 2)
	defer func() { cancel(); wait(); pool.Close() }()

	loaded := scoreP99(t, scorer, row, n)
	shap := medianContrib(t, pool, row, 21)

	_, declined := scorer.Counts()
	t.Logf("hot-path isolation: /score p99 baseline=%v loaded=%v | one SHAP=%v | declines=%d explained=%d",
		base, loaded, shap, declined, w.Explained())
	if declined != n {
		t.Fatalf("expected %d declines, got %d", n, declined)
	}
	if loaded >= shap/2 {
		t.Fatalf("loaded /score p99 %v approaches one SHAP %v - SHAP may be leaking onto the hot path", loaded, shap)
	}
}

// TestWorkerDrainOnQueueClose - примитив мягкого завершения: после остановки
// издателей и закрытия очереди пул воркеров дочищает буфер и выходит, поэтому ни
// одно отклонение в полёте не теряет своё объяснение.
func TestWorkerDrainOnQueueClose(t *testing.T) {
	pool := tdPool(t)
	row := make([]float64, pool.NumFeature())
	store := NewMemStore()
	queue := NewChannelQueue(64)
	w := NewWorker(pool, store, WorkerConfig{K: 3})
	wait := w.Start(context.Background(), queue.Events(), 4)

	const m = 30
	for i := 0; i < m; i++ {
		queue.Publish(DeclineEvent{ID: fmt.Sprintf("d%d", i), Row: row, Margin: -1})
	}
	queue.Close() // мягко: дочистить очередь, затем воркеры выходят
	wait()        // возвращается только после выхода каждого воркера
	pool.Close()

	for i := 0; i < m; i++ {
		if _, ok := store.Get(fmt.Sprintf("d%d", i)); !ok {
			t.Errorf("event d%d was not drained before shutdown", i)
		}
	}
}
