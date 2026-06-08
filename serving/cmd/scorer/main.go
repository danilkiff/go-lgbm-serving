// Команда scorer - сервис decline->explain. POST /score возвращает решение из
// пула Booster LightGBM. Режим -explain задаёт, где считается SHAP: async - для
// отклонений выкладывается DeclineEvent, воркеры считают SHAP вне горячего пути
// (под нагрузкой объяснение может быть отброшено при переполнении очереди);
// inline - SHAP считается на горячем пути для каждого решения и сохраняется до
// ответа (не теряется никогда, но каждый /score платит полный SHAP). GET
// /explain/{id} отдаёт объяснение, GET /metrics - операционный снимок.
//
//	scorer -model fixtures/model.txt -addr :8080 -threshold 0
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
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
	workers := flag.Int("workers", 2, "explain worker goroutines (explain=async)")
	storeKind := flag.String("store", "mem", "explanation store backend: mem | postgres")
	dsn := flag.String("dsn", "", "postgres DSN for -store=postgres (falls back to $SCORER_DSN)")
	explainMode := flag.String("explain", "async", "explanation path: async (off the hot path, may drop under load) | inline (computed on the hot path for every decision, never dropped)")
	flag.Parse()

	if *model == "" {
		log.Fatal("scorer: -model is required (e.g. -model fixtures/model.txt)")
	}

	n := runtime.GOMAXPROCS(0)
	hotPool, err := lgbm.NewPool(*model, n)
	if err != nil {
		log.Fatalf("scorer: load hot pool: %v", err)
	}
	var catalog *reasoncode.Catalog
	if *codes != "" {
		if catalog, err = reasoncode.LoadCatalog(*codes); err != nil {
			hotPool.Close()
			log.Fatalf("scorer: load codes: %v", err)
		}
	}
	store, closeStore, err := newStore(*storeKind, *dsn)
	if err != nil {
		hotPool.Close()
		log.Fatalf("scorer: %v", err)
	}

	// Горячий путь, снимок метрик и слив на завершении зависят от режима explain.
	var hot scorer
	var snapshot func() metricsResponse
	var drain func()

	switch *explainMode {
	case "inline":
		// Объяснение считается на горячем пути для КАЖДОГО решения и сохраняется
		// до ответа: ни очереди, ни воркеров - потерять его под нагрузкой нечему.
		// Цена - каждый /score платит полный SHAP (PredictContrib, примерно в
		// 46-58 раз дороже PredictRaw), а не только скоринг.
		sc := pipeline.NewInlineScorer(hotPool, *threshold, *model, store, *topk, catalog)
		hot = sc
		snapshot = func() metricsResponse {
			scored, declined := sc.Counts()
			m := metricsResponse{Scored: scored, Declined: declined, Explained: scored}
			if scored > 0 {
				m.DeclineRate = float64(declined) / float64(scored)
			}
			return m
		}
		drain = func() {}
		log.Printf("scorer: explain=inline, %d hot handles, store=%s, threshold=%g, top-%d reason codes, listening on %s", n, *storeKind, *threshold, *topk, *addr)

	case "async":
		// Горячий путь и explain владеют независимыми пулами хэндлов одной модели:
		// SHAP-воркеры физически не могут занять хэндлы скоринга (см.
		// TestHotPathIsolation). Цена - nWorkers лишних копий модели в памяти.
		nWorkers := max(*workers, 1)
		explainPool, perr := lgbm.NewPool(*model, nWorkers)
		if perr != nil {
			hotPool.Close()
			closeStore()
			log.Fatalf("scorer: load explain pool: %v", perr)
		}
		queue := pipeline.NewChannelQueue(*queueBuf)
		sc := pipeline.NewScorer(hotPool, *threshold, *model, queue, func(e pipeline.DeclineEvent) {
			log.Printf("score: decline id=%s dropped, explain queue full (queue_dropped=%d)", e.ID, queue.Dropped())
		})
		worker := pipeline.NewWorker(explainPool, store, pipeline.WorkerConfig{
			K:       *topk,
			Catalog: catalog,
			DeadLetter: func(e pipeline.DeclineEvent, err error) {
				log.Printf("explain: dead-letter id=%s: %v", e.ID, err)
			},
		})
		workerCtx, cancelWorkers := context.WithCancel(context.Background())
		waitWorkers := worker.Start(workerCtx, queue.Events(), nWorkers)
		hot = sc
		snapshot = func() metricsResponse {
			scored, declined := sc.Counts()
			m := metricsResponse{
				Scored: scored, Declined: declined,
				QueueLen: queue.Len(), QueueCap: queue.Cap(), QueueDropped: queue.Dropped(),
				Explained: worker.Explained(), DeadLettered: worker.Dropped(),
			}
			if scored > 0 {
				m.DeclineRate = float64(declined) / float64(scored)
			}
			return m
		}
		// Слить очередь: после Shutdown издателей нет -> воркеры дочищают остаток.
		drain = func() {
			queue.Close()
			done := make(chan struct{})
			go func() { waitWorkers(); close(done) }()
			select {
			case <-done:
			case <-time.After(10 * time.Second):
				log.Printf("scorer: explain drain timed out, %d events unprocessed", queue.Len())
				cancelWorkers()
				waitWorkers()
			}
			cancelWorkers()
			explainPool.Close()
		}
		log.Printf("scorer: explain=async, %d hot handles + %d explain handles, store=%s, threshold=%g, top-%d reason codes, listening on %s", n, nWorkers, *storeKind, *threshold, *topk, *addr)

	default:
		hotPool.Close()
		closeStore()
		log.Fatalf("scorer: unknown -explain %q (want async|inline)", *explainMode)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /score", scoreHandler(hot, maxScoreBody))
	mux.HandleFunc("GET /explain/{id}", explainHandler(store))
	mux.HandleFunc("GET /metrics", metricsHandler(snapshot))

	srv := &http.Server{Addr: *addr, Handler: mux}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	serveErr := make(chan error, 1)
	go func() { serveErr <- srv.ListenAndServe() }()

	select {
	case err := <-serveErr:
		if err != nil && err != http.ErrServerClosed {
			log.Printf("scorer: serve: %v", err)
		}
	case <-ctx.Done():
		log.Printf("scorer: signal received, draining...")
	}

	// Мягкое завершение: перестать принимать запросы (доделав текущие), затем (для
	// async) слить очередь explain, затем освободить хэндлы модели и хранилище.
	shutCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutCtx); err != nil {
		log.Printf("scorer: http shutdown: %v", err)
	}
	drain()
	hotPool.Close()
	closeStore()
	log.Printf("scorer: stopped")
}

// newStore выбирает бэкенд хранилища объяснений. mem - in-process MemStore;
// postgres - PgStore (DSN из -dsn или $SCORER_DSN). Возвращает хранилище и
// функцию его закрытия (для MemStore - no-op).
func newStore(kind, dsn string) (pipeline.Store, func(), error) {
	switch kind {
	case "mem":
		return pipeline.NewMemStore(), func() {}, nil
	case "postgres":
		if dsn == "" {
			dsn = os.Getenv("SCORER_DSN")
		}
		pg, err := pipeline.NewPgStore(context.Background(), dsn)
		if err != nil {
			return nil, nil, err
		}
		return pg, pg.Close, nil
	default:
		return nil, nil, fmt.Errorf("unknown -store %q (want mem|postgres)", kind)
	}
}

// scorer - поведение горячего пути, нужное HTTP-обработчику; *pipeline.Scorer
// его реализует, а в тестах - фейк (нативная либа не нужна).
type scorer interface {
	Score(row []float64) (pipeline.ScoreResult, error)
}

type scoreRequest struct {
	Features []float64 `json:"features"`
}

type scoreResponse struct {
	ID       string  `json:"id"`
	Margin   float64 `json:"margin"`
	Decision string  `json:"decision"`
}

// maxScoreBody - потолок размера тела POST /score. Вектор признаков - десятки
// float64; для недоверенного клиента это защита от чтения произвольно большого тела в память.
const maxScoreBody = 10240 // 10 KiB

func scoreHandler(s scorer, maxBody int64) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, maxBody)
		var req scoreRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			if _, ok := errors.AsType[*http.MaxBytesError](err); ok {
				http.Error(w, "request body too large", http.StatusRequestEntityTooLarge)
				return
			}
			http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
			return
		}
		res, err := s.Score(req.Features)
		if err != nil {
			http.Error(w, err.Error(), http.StatusUnprocessableEntity)
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

func metricsHandler(snapshot func() metricsResponse) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(snapshot())
	}
}
