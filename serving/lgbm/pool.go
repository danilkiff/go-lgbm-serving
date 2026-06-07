package lgbm

import "fmt"

// Pool - фиксированный набор независимых хэндлов Booster для ОДНОЙ модели,
// выдаваемых по одному на вызов. Это ответ на историю потокобезопасности C API
// LightGBM (#3751/#3771): вместо общего Booster на все горутины - где
// предсказание сериализуется внутренней блокировкой, а на версиях 3.0.0-3.1.1
// молча гонялось через thread-local буфер по ключу omp_get_thread_num() (для
// не-OpenMP потоков Go он возвращает 0) - каждая горутина берёт свой хэндл. Это
// даёт настоящий параллелизм без общего состояния предсказания.
//
// Цена: size копий модели в памяти.
type Pool struct {
	handles chan *Booster
	all     []*Booster
}

// NewPool загружает size независимых хэндлов из одного файла модели.
func NewPool(path string, size int) (*Pool, error) {
	if size < 1 {
		return nil, fmt.Errorf("lgbm: pool size must be >= 1, got %d", size)
	}
	p := &Pool{handles: make(chan *Booster, size), all: make([]*Booster, 0, size)}
	for i := 0; i < size; i++ {
		b, err := LoadBooster(path)
		if err != nil {
			p.Close()
			return nil, fmt.Errorf("lgbm: pool handle %d: %w", i, err)
		}
		p.all = append(p.all, b)
		p.handles <- b
	}
	return p, nil
}

// NumFeature возвращает число входных признаков модели.
func (p *Pool) NumFeature() int { return p.all[0].nFeature }

// PredictRaw берёт хэндл, считает сырую маржу и возвращает её.
func (p *Pool) PredictRaw(row []float64) (float64, error) {
	b := <-p.handles
	defer func() { p.handles <- b }()
	return b.PredictRaw(row)
}

// PredictContrib берёт хэндл и возвращает нативные вклады SHAP.
func (p *Pool) PredictContrib(row []float64) ([]float64, error) {
	b := <-p.handles
	defer func() { p.handles <- b }()
	return b.PredictContrib(row)
}

// Close освобождает все хэндлы. Небезопасно вызывать конкурентно с Predict*.
func (p *Pool) Close() {
	for _, b := range p.all {
		b.Close()
	}
	p.all = nil
}
