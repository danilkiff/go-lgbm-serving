// Пакет pipeline - топология подачи decline->explain. Горячий путь скорит
// транзакцию через пул Booster LightGBM и только для отклонений выкладывает
// DeclineEvent вне горячего пути; нативные коды причин SHAP считает воркер
// асинхронно, никогда не инлайн - именно эту стоимость (примерно в 40 раз дороже
// скоринга) мы держим вне критического пути.
//
// Этот файл - половина горячего пути: Decision, DeclineEvent, контракт Queue с
// дефолтной in-process реализацией и Scorer.
package pipeline

import (
	"crypto/rand"
	"encoding/hex"
	"sync/atomic"

	"github.com/danilkiff/go-lgbm-serving/lgbm"
)

// Decision - вердикт горячего пути по одной транзакции.
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
// поэтому оно не может разойтись с поданным решением.
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
	dropped int64
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
		atomic.AddInt64(&q.dropped, 1)
		return false
	}
}

// Events - сторона потребителя очереди.
func (q *ChannelQueue) Events() <-chan DeclineEvent { return q.ch }

// Dropped сообщает, сколько событий отброшено из-за полной очереди.
func (q *ChannelQueue) Dropped() int64 { return atomic.LoadInt64(&q.dropped) }

// Len - число событий в буфере прямо сейчас (глубина очереди).
func (q *ChannelQueue) Len() int { return len(q.ch) }

// Cap - ёмкость буфера очереди.
func (q *ChannelQueue) Cap() int { return cap(q.ch) }

// Close закрывает очередь, чтобы воркеры, читающие Events(), получили
// накопленный остаток и затем остановились. Вызывать только после остановки всех
// издателей (например, после остановки HTTP-сервера) - publish после Close
// паникует.
func (q *ChannelQueue) Close() { close(q.ch) }

// ScoreResult - ответ горячего пути по одной транзакции.
type ScoreResult struct {
	ID       string
	Margin   float64
	Decision Decision
}

// Scorer - горячий путь: скоринг -> решение -> (для отклонений) публикация
// DeclineEvent. SHAP никогда не считается инлайн.
type Scorer struct {
	pool      *lgbm.Pool
	threshold float64
	modelVer  string
	queue     Queue
	newID     func() string
	scored    int64
	declined  int64
}

// NewScorer собирает горячий путь. Транзакция отклоняется, когда её сырая маржа
// превышает threshold. queue может быть nil - тогда отклонения ничего не
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
	atomic.AddInt64(&s.scored, 1)
	res := ScoreResult{ID: s.newID(), Margin: margin, Decision: Approve}
	if margin > s.threshold {
		res.Decision = Decline
		atomic.AddInt64(&s.declined, 1)
		if s.queue != nil {
			// Копируем строку: вызывающий может переиспользовать срез, а событие
			// живёт дольше этого вызова.
			row2 := append([]float64(nil), row...)
			s.queue.Publish(DeclineEvent{ID: res.ID, Row: row2, Margin: margin, ModelVer: s.modelVer})
		}
	}
	return res, nil
}

// Counts возвращает, сколько транзакций сосчитано и сколько отклонено.
func (s *Scorer) Counts() (scored, declined int64) {
	return atomic.LoadInt64(&s.scored), atomic.LoadInt64(&s.declined)
}

func randID() string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}
