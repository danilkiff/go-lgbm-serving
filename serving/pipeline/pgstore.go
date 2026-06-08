package pipeline

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"sync/atomic"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// PgStore - устойчивый Store объяснений поверх PostgreSQL: тот же путь explain,
// что и у MemStore, но артефакт переживает рестарт процесса. Запись идёт из
// воркеров explain, чтение - из GET /explain/{id}; горячий путь /score хранилище
// не трогает. Пул соединений pgx безопасен для конкурентного использования.
type PgStore struct {
	pool   *pgxpool.Pool
	putErr atomic.Int64
}

// pgSchema идемпотентен: id решения - первичный ключ, reasons лежат как jsonb.
const pgSchema = `
CREATE TABLE IF NOT EXISTS explanations (
	id         text PRIMARY KEY,
	margin     double precision NOT NULL,
	base       double precision NOT NULL,
	model_ver  text NOT NULL,
	reasons    jsonb NOT NULL,
	created_at timestamptz NOT NULL DEFAULT now()
);`

// pgOpTimeout - потолок на один Put/Get/Ping, чтобы зависшее соединение не
// держало ни воркер explain, ни обработчик GET.
const pgOpTimeout = 5 * time.Second

// NewPgStore подключается по DSN, дожидается готовности postgres (он мог ещё
// подниматься) и накатывает идемпотентную схему.
func NewPgStore(ctx context.Context, dsn string) (*PgStore, error) {
	if dsn == "" {
		return nil, fmt.Errorf("pgstore: empty DSN (set -dsn or $SCORER_DSN)")
	}
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("pgstore: connect: %w", err)
	}
	if err := pingWithRetry(ctx, pool); err != nil {
		pool.Close()
		return nil, fmt.Errorf("pgstore: ping: %w", err)
	}
	if _, err := pool.Exec(ctx, pgSchema); err != nil {
		pool.Close()
		return nil, fmt.Errorf("pgstore: schema: %w", err)
	}
	return &PgStore{pool: pool}, nil
}

// pingWithRetry пингует пул до 30 секунд: контейнер postgres стартует не мгновенно.
func pingWithRetry(ctx context.Context, pool *pgxpool.Pool) error {
	var err error
	for range 60 {
		c, cancel := context.WithTimeout(ctx, pgOpTimeout)
		err = pool.Ping(c)
		cancel()
		if err == nil {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}
	}
	return err
}

// Put сохраняет объяснение по e.ID. Store.Put не возвращает ошибку (MemStore не
// падает); сбой записи в pg логируется и учитывается (PutErrors) - симметрично
// dead-letter воркера, чтобы потеря объяснения была видна, а не молчала. id
// детерминирует объяснение, поэтому повторная вставка - DO NOTHING.
func (s *PgStore) Put(e Explanation) {
	reasons, err := json.Marshal(e.Reasons)
	if err != nil {
		s.putErr.Add(1)
		log.Printf("pgstore: marshal id=%s: %v", e.ID, err)
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), pgOpTimeout)
	defer cancel()
	_, err = s.pool.Exec(ctx,
		`INSERT INTO explanations (id, margin, base, model_ver, reasons)
		 VALUES ($1, $2, $3, $4, $5)
		 ON CONFLICT (id) DO NOTHING`,
		e.ID, e.Margin, e.Base, e.ModelVer, reasons)
	if err != nil {
		s.putErr.Add(1)
		log.Printf("pgstore: put id=%s: %v", e.ID, err)
	}
}

// Get возвращает объяснение для id или false, если его ещё нет (как и у MemStore,
// согласованность в конечном счёте: воркер мог не успеть записать).
func (s *PgStore) Get(id string) (Explanation, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), pgOpTimeout)
	defer cancel()
	e := Explanation{ID: id}
	var reasons []byte
	err := s.pool.QueryRow(ctx,
		`SELECT margin, base, model_ver, reasons FROM explanations WHERE id = $1`, id).
		Scan(&e.Margin, &e.Base, &e.ModelVer, &reasons)
	if err != nil {
		if !errors.Is(err, pgx.ErrNoRows) {
			log.Printf("pgstore: get id=%s: %v", id, err)
		}
		return Explanation{}, false
	}
	if err := json.Unmarshal(reasons, &e.Reasons); err != nil {
		log.Printf("pgstore: unmarshal id=%s: %v", id, err)
		return Explanation{}, false
	}
	return e, true
}

// PutErrors сообщает, сколько записей объяснений провалилось (для операционной видимости).
func (s *PgStore) PutErrors() int64 { return s.putErr.Load() }

// Close освобождает пул соединений.
func (s *PgStore) Close() { s.pool.Close() }
