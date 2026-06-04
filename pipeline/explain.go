package pipeline

import (
	"context"
	"sync"
	"sync/atomic"

	"github.com/danilkiff/go-lgbm-serving/lgbm"
	"github.com/danilkiff/go-lgbm-serving/reasoncode"
)

// ReasonCode - один ранжированный вклад в решение: признак, его код/метка
// adverse-action (через reasoncode.Catalog), толкнул ли он к отклонению или от
// него, и знаковый вклад SHAP.
type ReasonCode struct {
	Feature      int     `json:"feature"`
	Code         string  `json:"code"`
	Label        string  `json:"label"`
	Direction    string  `json:"direction"`
	Contribution float64 `json:"contribution"`
}

// Explanation - артефакт вне пути для отклонённой транзакции: топ-K кодов причин
// плюс значения, по которым его можно сверить с поданным решением (Margin) и
// моделью, его принявшей (ModelVer).
type Explanation struct {
	ID       string       `json:"id"`
	Margin   float64      `json:"margin"`
	Base     float64      `json:"base"`
	Reasons  []ReasonCode `json:"reasons"`
	ModelVer string       `json:"model_ver"`
}

// Store сохраняет и достаёт объяснения по id решения. Дефолтный MemStore
// работает in-process; устойчивое хранилище - более поздний адаптер.
type Store interface {
	Put(Explanation)
	Get(id string) (Explanation, bool)
}

// MemStore - in-memory Store, безопасный для конкурентного использования.
type MemStore struct {
	mu sync.RWMutex
	m  map[string]Explanation
}

// NewMemStore возвращает пустое in-memory хранилище.
func NewMemStore() *MemStore { return &MemStore{m: make(map[string]Explanation)} }

// Put сохраняет e по ключу e.ID.
func (s *MemStore) Put(e Explanation) {
	s.mu.Lock()
	s.m[e.ID] = e
	s.mu.Unlock()
}

// Get возвращает объяснение для id или false, если оно ещё не сохранено.
func (s *MemStore) Get(id string) (Explanation, bool) {
	s.mu.RLock()
	e, ok := s.m[id]
	s.mu.RUnlock()
	return e, ok
}

// WorkerConfig настраивает воркер explain.
type WorkerConfig struct {
	// K - сколько верхних кодов причин хранить на объяснение.
	K int
	// Catalog размечает признаки кодами adverse-action (nil -> обобщённые коды).
	Catalog *reasoncode.Catalog
	// Retries - сколько дополнительных попыток PredictContrib делать при сбое
	// (0 = одна попытка), защита от временных нативных ошибок.
	Retries int
	// DeadLetter получает событие, чьё объяснение так и не удалось после всех
	// попыток, чтобы сбои были видны, а не молча терялись. nil -> событие лишь
	// учитывается (см. Dropped).
	DeadLetter func(DeclineEvent, error)
}

// Worker вычёрпывает DeclineEvent вне горячего пути, считает нативный SHAP для
// каждого через пул Booster, ранжирует топ-K кодов причин и сохраняет результат.
// Здесь и живёт стоимость SHAP (примерно в 40 раз дороже скоринга) - никогда на
// пути /score. Worker безопасно запускать пулом горутин (см. Start).
type Worker struct {
	pool      *lgbm.Pool
	store     Store
	cfg       WorkerConfig
	explained int64
	dropped   int64
}

// NewWorker возвращает воркер explain.
func NewWorker(pool *lgbm.Pool, store Store, cfg WorkerConfig) *Worker {
	return &Worker{pool: pool, store: store, cfg: cfg}
}

// Run потребляет события, пока канал не закроется или ctx не отменят. Безопасен
// для конкурентного запуска пулом горутин (см. Start).
func (w *Worker) Run(ctx context.Context, events <-chan DeclineEvent) {
	for {
		select {
		case <-ctx.Done():
			return
		case e, ok := <-events:
			if !ok {
				return
			}
			w.process(e)
		}
	}
}

// Start запускает n горутин-воркеров, читающих events, и возвращает функцию,
// блокирующую до выхода их всех (после отмены ctx или закрытия events).
func (w *Worker) Start(ctx context.Context, events <-chan DeclineEvent, n int) func() {
	if n < 1 {
		n = 1
	}
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			w.Run(ctx, events)
		}()
	}
	return wg.Wait
}

// process считает и сохраняет одно объяснение, повторяя временные сбои и отправляя
// в dead-letter событие, которое всё равно падает, - так код причины отклонения
// никогда не теряется молча.
func (w *Worker) process(e DeclineEvent) {
	var err error
	for attempt := 0; attempt <= w.cfg.Retries; attempt++ {
		var exp Explanation
		if exp, err = w.explain(e); err == nil {
			w.store.Put(exp)
			atomic.AddInt64(&w.explained, 1)
			return
		}
	}
	atomic.AddInt64(&w.dropped, 1)
	if w.cfg.DeadLetter != nil {
		w.cfg.DeadLetter(e, err)
	}
}

// Explained сообщает, сколько объяснений посчитано и сохранено.
func (w *Worker) Explained() int64 { return atomic.LoadInt64(&w.explained) }

// Dropped сообщает, сколько событий провалили все попытки (и ушли в dead-letter).
func (w *Worker) Dropped() int64 { return atomic.LoadInt64(&w.dropped) }

// explain считает ранжированные коды причин для одного события отклонения.
// Вклады берутся из того же нативного предиктора, что дал поданную маржу, поэтому
// инвариант sum(contrib) == margin связывает объяснение с решением.
func (w *Worker) explain(e DeclineEvent) (Explanation, error) {
	contrib, err := w.pool.PredictContrib(e.Row)
	if err != nil {
		return Explanation{}, err
	}
	nf := len(contrib) - 1 // последний элемент - базовое (ожидаемое) значение
	top := reasoncode.TopK(contrib[:nf], w.cfg.K)
	reasons := make([]ReasonCode, len(top))
	for i, idx := range top {
		code := w.cfg.Catalog.Lookup(idx)
		reasons[i] = ReasonCode{
			Feature:      idx,
			Code:         code.Code,
			Label:        code.Label,
			Direction:    reasoncode.Direction(contrib[idx]),
			Contribution: contrib[idx],
		}
	}
	return Explanation{
		ID:       e.ID,
		Margin:   e.Margin,
		Base:     contrib[nf],
		Reasons:  reasons,
		ModelVer: e.ModelVer,
	}, nil
}
