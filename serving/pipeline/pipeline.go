// Пакет pipeline - топология конвейера decline->explain. Горячий путь скорит
// попытку входа через пул Booster LightGBM и только для отклонений выкладывает
// DeclineEvent вне горячего пути; нативные коды причин SHAP считает воркер
// асинхронно, никогда не инлайн - именно эту стоимость (в десятки раз дороже
// скоринга; замеры - в README) мы держим вне критического пути.
package pipeline

import (
	"crypto/rand"
	"encoding/hex"
	"slices"
	"sync/atomic"

	"github.com/danilkiff/go-lgbm-serving/lgbm"
)

// Decision - вердикт горячего пути по одной попытке входа.
type Decision uint8

const (
	Approve Decision = iota
	Decline
)

func (d Decision) String() string {
	if d == Decline {
		return "decline"
	}
	return "approve"
}

// DeclineEvent выкладывается вне горячего пути на каждое отклонение. Воркер
// explain потребляет его, считает нативный SHAP и сохраняет коды причин. Row и
// ModelVer позволяют посчитать объяснение той же моделью, что приняла решение,
// поэтому оно не может разойтись с принятым решением.
type DeclineEvent struct {
	ID       string
	Row      []float64
	Margin   float64
	ModelVer string
}

// Queue несёт DeclineEvent с горячего пути к воркеру explain. Дефолтная
// ChannelQueue работает in-process; внешний брокер - более поздний адаптер за
// тем же интерфейсом.
type Queue interface {
	// Publish никогда не должен блокировать горячий путь. Возвращает false, если
	// событие отброшено (очередь полна), чтобы вызывающий мог учесть потерю.
	Publish(DeclineEvent) bool
	// Events - сторона потребителя, которую вычерпывает воркер.
	Events() <-chan DeclineEvent
}

// ChannelQueue - ограниченная in-process очередь. Publish неблокирующий: при
// полном буфере событие отбрасывается и учитывается, поэтому медленный или
// отсутствующий потребитель не добавит задержки горячему пути.
type ChannelQueue struct {
	ch      chan DeclineEvent
	dropped atomic.Int64
	// OnDrop, если задан, вызывается на каждое отброшенное событие: отброс - это
	// отклонение, навсегда оставшееся без объяснения, и след по id обязан быть
	// per-event, а не только счётчиком. Выполняется на горячем пути издателя -
	// только дешёвые действия. Задавать до первого Publish.
	OnDrop func(DeclineEvent)
}

// NewChannelQueue возвращает ChannelQueue с буфером на buffer событий.
func NewChannelQueue(buffer int) *ChannelQueue {
	if buffer < 0 {
		buffer = 0
	}
	return &ChannelQueue{ch: make(chan DeclineEvent, buffer)}
}

// Publish ставит e в очередь без блокировки; возвращает false, если буфер полон.
func (q *ChannelQueue) Publish(e DeclineEvent) bool {
	select {
	case q.ch <- e:
		return true
	default:
		q.dropped.Add(1)
		if q.OnDrop != nil {
			q.OnDrop(e)
		}
		return false
	}
}

// Events - сторона потребителя очереди.
func (q *ChannelQueue) Events() <-chan DeclineEvent { return q.ch }

// Dropped сообщает, сколько событий отброшено из-за полной очереди.
func (q *ChannelQueue) Dropped() int64 { return q.dropped.Load() }

// Len - число событий в буфере прямо сейчас (глубина очереди).
func (q *ChannelQueue) Len() int { return len(q.ch) }

// Cap - ёмкость буфера очереди.
func (q *ChannelQueue) Cap() int { return cap(q.ch) }

// Close закрывает очередь, чтобы воркеры, читающие Events(), получили
// накопленный остаток и затем остановились. Вызывать только после остановки всех
// издателей (например, после остановки HTTP-сервера) - publish после Close
// паникует.
func (q *ChannelQueue) Close() { close(q.ch) }

// ScoreResult - ответ горячего пути по одной попытке входа. Объяснение
// best-effort: ExplainQueued=false у одобрений и у отклонений, чьё событие
// отброшено полной очередью, - вызывающий узнаёт о потере сразу, а не вечным
// 404 на /explain.
type ScoreResult struct {
	ID            string
	Margin        float64
	Decision      Decision
	ExplainQueued bool
}

// Scorer - горячий путь: скоринг попытки входа -> решение -> (для отклонений)
// публикация DeclineEvent. SHAP никогда не считается инлайн.
type Scorer struct {
	pool      *lgbm.Pool
	threshold float64
	modelVer  string
	queue     Queue
	newID     func() string
	scored    atomic.Int64
	declined  atomic.Int64
}

// NewScorer собирает горячий путь. Попытка входа отклоняется, когда её raw
// margin превышает threshold. queue может быть nil - тогда отклонения ничего не
// выкладывают.
func NewScorer(pool *lgbm.Pool, threshold float64, modelVer string, queue Queue) *Scorer {
	return &Scorer{
		pool:      pool,
		threshold: threshold,
		modelVer:  modelVer,
		queue:     queue,
		newID:     randID,
	}
}

// Score прогоняет строку по горячему пути и при отклонении выкладывает
// DeclineEvent вне пути.
func (s *Scorer) Score(row []float64) (ScoreResult, error) {
	margin, err := s.pool.PredictRaw(row)
	if err != nil {
		return ScoreResult{}, err
	}
	s.scored.Add(1)
	res := ScoreResult{ID: s.newID(), Margin: margin, Decision: Approve}
	if margin > s.threshold {
		res.Decision = Decline
		s.declined.Add(1)
		if s.queue != nil {
			// Копируем строку: вызывающий может переиспользовать срез, а событие
			// живёт дольше этого вызова.
			event := DeclineEvent{
				ID:       res.ID,
				Row:      slices.Clone(row),
				Margin:   margin,
				ModelVer: s.modelVer,
			}
			res.ExplainQueued = s.queue.Publish(event)
		}
	}
	return res, nil
}

// Counts возвращает, сколько попыток входа сосчитано и сколько отклонено.
func (s *Scorer) Counts() (scored, declined int64) {
	return s.scored.Load(), s.declined.Load()
}

func randID() string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}
