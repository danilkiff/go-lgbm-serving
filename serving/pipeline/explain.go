package pipeline

import (
	"context"
	"sync"
	"sync/atomic"

	"github.com/danilkiff/go-lgbm-serving/lgbm"
	"github.com/danilkiff/go-lgbm-serving/reasoncode"
)

// ReasonCode - один ранжированный contribution в решение: признак, его код/метка
// adverse-action (через reasoncode.Catalog) и знаковый SHAP contribution. В
// Reasons отбираются только толкавшие к отклонению (Direction у них всегда
// "increased risk"; поле оставлено, чтобы артефакт читался сам по себе, без
// знания правила отбора).
type ReasonCode struct {
	Feature      int     `json:"feature"`
	Code         string  `json:"code"`
	Label        string  `json:"label"`
	Direction    string  `json:"direction"`
	Contribution float64 `json:"contribution"`
}

// Explanation - артефакт вне пути для отклонённой попытки входа: топ-K кодов причин
// плюс значения, по которым его можно сверить с принятым решением (Margin) и
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
// Растёт неограниченно: записи не вытесняются, пока жив процесс - долгоживущему
// сервису нужен адаптер с retention-политикой за тем же интерфейсом Store.
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
	// DeadLetter получает событие, чьё объяснение не удалось, чтобы сбои были
	// видны, а не молча терялись. nil -> событие лишь учитывается (см. Dropped).
	DeadLetter func(DeclineEvent, error)
}

// Worker вычёрпывает DeclineEvent вне горячего пути, считает нативный SHAP для
// каждого через пул Booster, ранжирует топ-K кодов причин и сохраняет результат.
// Здесь и живёт стоимость SHAP (в десятки раз дороже скоринга) - никогда на
// пути /score. Worker безопасно запускать пулом горутин (см. Start).
type Worker struct {
	pool      *lgbm.Pool
	store     Store
	cfg       WorkerConfig
	explained atomic.Int64
	dropped   atomic.Int64
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

// process считает и сохраняет одно объяснение; сбой уходит в dead-letter - так
// код причины отклонения никогда не теряется молча. Повторов нет: вход и модель
// в памяти те же, нативный сбой детерминирован, и повтор ничего не добавлял бы.
func (w *Worker) process(e DeclineEvent) {
	exp, err := w.explain(e)
	if err != nil {
		w.dropped.Add(1)
		if w.cfg.DeadLetter != nil {
			w.cfg.DeadLetter(e, err)
		}
		return
	}
	w.store.Put(exp)
	w.explained.Add(1)
}

// Explained сообщает, сколько объяснений посчитано и сохранено.
func (w *Worker) Explained() int64 { return w.explained.Load() }

// Dropped сообщает, сколько событий не удалось объяснить (ушли в dead-letter).
func (w *Worker) Dropped() int64 { return w.dropped.Load() }

// explain считает ранжированные коды причин для одного события отклонения.
// Contributions берутся из того же нативного предиктора, что дал сам margin,
// поэтому инвариант sum(contrib) == margin связывает объяснение с решением по
// построению: один пул из одних байт, margin копируется в событие дословно.
// Закреплён инвариант паритетным и e2e тестами; runtime-сверки нет намеренно -
// в однопроцессной топологии она лишь добавляла бы новый путь потери
// объяснения, не закрывая реального рассогласования. Понадобится она, когда
// события начнёт приносить внешний Queue-адаптер из другого процесса.
func (w *Worker) explain(e DeclineEvent) (Explanation, error) {
	contrib, err := w.pool.PredictContrib(e.Row)
	if err != nil {
		return Explanation{}, err
	}
	nf := len(contrib) - 1 // последний элемент - base value
	// Причина отклонения - только contribution, толкавший к нему: отрицательные
	// (аргументы за approve) и нулевые в Reasons не попадают, поэтому список
	// бывает короче K.
	top := reasoncode.TopKPositive(contrib[:nf], w.cfg.K)
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
