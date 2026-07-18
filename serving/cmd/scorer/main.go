// Команда scorer - сервис decline->explain. POST /score возвращает решение из
// пула Booster LightGBM и для отклонений выкладывает DeclineEvent вне горячего
// пути; шаг SHAP (в десятки раз дороже скоринга) никогда не идёт инлайн.
// Воркеры explain считают SHAP асинхронно, GET /explain/{id} отдаёт результат,
// GET /metrics - операционный снимок.
//
//	scorer -model fixtures/model.txt -addr :8080 -threshold 0
package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/danilkiff/go-lgbm-serving/lgbm"
	"github.com/danilkiff/go-lgbm-serving/pipeline"
	"github.com/danilkiff/go-lgbm-serving/reasoncode"
)

func main() {
	model := flag.String("model", "", "LightGBM model file (required, e.g. fixtures/model.txt)")
	addr := flag.String("addr", ":8080", "listen address")
	threshold := flag.Float64("threshold", 0, "raw-margin decline threshold (decline if margin > threshold)")
	queueBuf := flag.Int("queue", 1024, "decline-event queue buffer")
	topk := flag.Int("topk", 3, "reason codes per explanation")
	codes := flag.String("codes", "", "optional JSON file mapping feature index to adverse-action {code,label}")
	workers := flag.Int("workers", 2, "explain worker goroutines (share the Booster pool)")
	flag.Parse()

	if *model == "" {
		log.Fatal("scorer: -model is required (e.g. -model fixtures/model.txt)")
	}
	// Версия модели - путь плюс sha256-префикс файла: объяснение в Explanation
	// привязано к байтам модели, принявшей решение, а не к имени файла.
	modelVer, err := modelVersion(*model)
	if err != nil {
		log.Fatalf("scorer: model version: %v", err)
	}

	// Хэндлов больше, чем воркеров explain: даже при всех воркерах, занятых SHAP,
	// горячему пути остаётся GOMAXPROCS хэндлов. Резерв закрывает голод по
	// хэндлам, но не контентию по CPU (см. TestHotPathIsolation).
	n := runtime.GOMAXPROCS(0) + *workers
	pool, err := lgbm.NewPool(*model, n)
	if err != nil {
		log.Fatalf("scorer: load pool: %v", err)
	}
	var catalog *reasoncode.Catalog
	if *codes != "" {
		if catalog, err = reasoncode.LoadCatalog(*codes); err != nil {
			pool.Close()
			log.Fatalf("scorer: load codes: %v", err)
		}
	}

	queue := pipeline.NewChannelQueue(*queueBuf)
	// Переполнение очереди - конкретное отклонение навсегда без объяснения;
	// агрегатного queue_dropped мало, потеря логируется по id.
	queue.OnDrop = func(e pipeline.DeclineEvent) {
		log.Printf("explain: queue full, explanation lost id=%s margin=%g", e.ID, e.Margin)
	}
	store := pipeline.NewMemStore()
	scorer := pipeline.NewScorer(pool, *threshold, modelVer, queue)
	worker := pipeline.NewWorker(pool, store, pipeline.WorkerConfig{
		K:       *topk,
		Catalog: catalog,
		DeadLetter: func(e pipeline.DeclineEvent, err error) {
			log.Printf("explain: dead-letter id=%s: %v", e.ID, err)
		},
	})

	// Воркеры explain делят пул Booster с горячим путём; стоимость SHAP считается
	// здесь, асинхронно, и на /score не попадает.
	workerCtx, cancelWorkers := context.WithCancel(context.Background())
	defer cancelWorkers()
	waitWorkers := worker.Start(workerCtx, queue.Events(), *workers)

	mux := http.NewServeMux()
	mux.HandleFunc("POST /score", scoreHandler(scorer))
	mux.HandleFunc("GET /explain/{id}", explainHandler(store))
	mux.HandleFunc("GET /metrics", func(w http.ResponseWriter, r *http.Request) {
		scored, declined := scorer.Counts()
		m := metricsResponse{
			Scored: scored, Declined: declined,
			QueueLen: queue.Len(), QueueCap: queue.Cap(), QueueDropped: queue.Dropped(),
			Explained: worker.Explained(), DeadLettered: worker.Dropped(),
		}
		if scored > 0 {
			m.DeclineRate = float64(declined) / float64(scored)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(m)
	})

	srv := &http.Server{Addr: *addr, Handler: mux}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	serveErr := make(chan error, 1)
	go func() { serveErr <- srv.ListenAndServe() }()
	log.Printf("scorer: %d handles, %d explain workers, threshold=%g, top-%d reason codes, listening on %s", n, *workers, *threshold, *topk, *addr)

	select {
	case err := <-serveErr:
		if err != nil && err != http.ErrServerClosed {
			log.Printf("scorer: serve: %v", err)
		}
	case <-ctx.Done():
		log.Printf("scorer: signal received, draining...")
	}

	// Мягкое завершение, по порядку: перестать принимать запросы (доделав
	// текущие), слить очередь explain, затем освободить хэндлы модели.
	shutCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutCtx); err != nil {
		// Таймаут Shutdown: обработчики ещё в полёте - могут публиковать в очередь
		// (Publish в закрытый канал паникует) и держать хэндлы пула (Close под
		// активным Predict - use-after-free). Безопасно только выйти.
		log.Printf("scorer: http shutdown: %v", err)
		os.Exit(1)
	}
	queue.Close() // Shutdown успешен: издателей нет -> воркеры дочищают очередь
	drained := make(chan struct{})
	go func() { waitWorkers(); close(drained) }()
	select {
	case <-drained:
	case <-shutCtx.Done():
		log.Printf("scorer: explain drain timed out, %d events unprocessed", queue.Len())
		cancelWorkers()
		waitWorkers()
	}
	pool.Close()
	log.Printf("scorer: stopped")
}

// modelVersion возвращает "путь@sha256-префикс" файла модели - идентификатор,
// по которому объяснение сверяется с байтами модели, а не с именем файла.
func modelVersion(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return fmt.Sprintf("%s@%x", path, sum[:8]), nil
}

// scorer - поведение горячего пути, нужное HTTP-обработчику; *pipeline.Scorer
// его реализует, а в тестах - фейк (нативная либа не нужна).
type scorer interface {
	Score(row []float64) (pipeline.ScoreResult, error)
}

type scoreRequest struct {
	// Указатели ради null: JSON не умеет NaN, а missing-значения (непомеренный
	// RTT - штатный случай, ~96% в обучающих данных) идут в missing-ветки
	// деревьев. null -> NaN; молчаливый ноль был бы другим, легитимным значением.
	Features []*float64 `json:"features"`
}

type scoreResponse struct {
	ID       string  `json:"id"`
	Margin   float64 `json:"margin"`
	Decision string  `json:"decision"`
}

func scoreHandler(s scorer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Валидный запрос - сотни байт (12 float64); мегабайт отсекает только
		// мусор, не давая раздуть память сервиса произвольным телом.
		r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
		var req scoreRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			var tooLarge *http.MaxBytesError
			if errors.As(err, &tooLarge) {
				http.Error(w, err.Error(), http.StatusRequestEntityTooLarge)
				return
			}
			http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
			return
		}
		row := make([]float64, len(req.Features))
		for i, f := range req.Features {
			if f == nil {
				row[i] = math.NaN()
			} else {
				row[i] = *f
			}
		}
		res, err := s.Score(row)
		if err != nil {
			// Неверная ширина входа - ошибка клиента (422); всё прочее - сбой
			// нативного предиктора, и это 500, а не вина запроса.
			status := http.StatusInternalServerError
			if errors.Is(err, lgbm.ErrFeatureCount) {
				status = http.StatusUnprocessableEntity
			}
			http.Error(w, err.Error(), status)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(scoreResponse{
			ID:       res.ID,
			Margin:   res.Margin,
			Decision: res.Decision.String(),
		})
	}
}

// explainer - поиск, нужный обработчику /explain; *pipeline.MemStore его
// реализует, а в тестах - фейк.
type explainer interface {
	Get(id string) (pipeline.Explanation, bool)
}

func explainHandler(e explainer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		exp, ok := e.Get(r.PathValue("id"))
		if !ok {
			// Согласованность в конечном счёте: только что отклонённый id может
			// быть ещё не объяснён.
			http.Error(w, "explanation not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(exp)
	}
}

// metricsResponse - операционный снимок для GET /metrics: счётчики горячего
// пути, очередь explain и прогресс объяснителя.
type metricsResponse struct {
	Scored       int64   `json:"scored"`
	Declined     int64   `json:"declined"`
	DeclineRate  float64 `json:"decline_rate"`
	QueueLen     int     `json:"queue_len"`
	QueueCap     int     `json:"queue_cap"`
	QueueDropped int64   `json:"queue_dropped"`
	Explained    int64   `json:"explained"`
	DeadLettered int64   `json:"dead_lettered"`
}
