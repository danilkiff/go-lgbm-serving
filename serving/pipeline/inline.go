package pipeline

import (
	"sync/atomic"

	"github.com/danilkiff/go-lgbm-serving/lgbm"
	"github.com/danilkiff/go-lgbm-serving/reasoncode"
)

// InlineScorer считает объяснение на горячем пути для КАЖДОГО решения (принятого и
// отклонённого) и сохраняет его синхронно до ответа. В отличие от связки Scorer +
// Worker здесь нет ни очереди, ни воркеров: объяснение не может потеряться под
// нагрузкой (нечему переполняться). Цена - каждый /score платит полную стоимость
// нативного SHAP (PredictContrib, примерно в 46-58 раз дороже PredictRaw), а не
// только скоринга, плюс синхронную запись в Store. Размен: пропускная способность
// и латентность против гарантии, что у каждого обслуженного решения есть
// объяснение, а перегрузка проявляется явно (медленный/отклонённый запрос), а не
// тихой потерей объяснения.
type InlineScorer struct {
	pool      *lgbm.Pool
	threshold float64
	modelVer  string
	store     Store
	k         int
	catalog   *reasoncode.Catalog
	newID     func() string
	scored    atomic.Int64
	declined  atomic.Int64
}

// NewInlineScorer собирает синхронный горячий путь со встроенным объяснением.
func NewInlineScorer(pool *lgbm.Pool, threshold float64, modelVer string, store Store, k int, catalog *reasoncode.Catalog) *InlineScorer {
	return &InlineScorer{
		pool:      pool,
		threshold: threshold,
		modelVer:  modelVer,
		store:     store,
		k:         k,
		catalog:   catalog,
		newID:     randID,
	}
}

// Score считает нативный SHAP, выводит из него margin и решение, строит топ-K
// кодов причин и сохраняет объяснение - всё на горячем пути, до возврата. margin
// здесь - сумма contributions (инвариант sum(contrib) == raw margin): решение и
// объяснение исходят из одного предиктора и не могут разойтись.
func (s *InlineScorer) Score(row []float64) (ScoreResult, error) {
	contrib, err := s.pool.PredictContrib(row)
	if err != nil {
		return ScoreResult{}, err
	}
	s.scored.Add(1)
	margin := 0.0
	for _, c := range contrib {
		margin += c
	}
	res := ScoreResult{ID: s.newID(), Margin: margin, Decision: Approve}
	if margin > s.threshold {
		res.Decision = Decline
		s.declined.Add(1)
	}
	s.store.Put(Explanation{
		ID:       res.ID,
		Margin:   margin,
		Base:     contrib[len(contrib)-1],
		Reasons:  buildReasons(contrib, s.k, s.catalog),
		ModelVer: s.modelVer,
	})
	return res, nil
}

// Counts возвращает, сколько решений сосчитано и сколько отклонено.
func (s *InlineScorer) Counts() (scored, declined int64) {
	return s.scored.Load(), s.declined.Load()
}
