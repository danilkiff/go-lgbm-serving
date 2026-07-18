package lgbm

import (
	"fmt"
	"os"
)

// Pool - фиксированный набор независимых хэндлов Booster одной модели, по одному
// на вызов: каждая горутина берёт свой хэндл, отсюда настоящий параллелизм без
// общего состояния предсказания. Ответ на потокобезопасность C API LightGBM
// (#3751/#3771): общий Booster сериализует предсказание блокировкой, а на
// 3.0.0-3.1.1 молча гонялся через thread-local буфер по omp_get_thread_num() (для
// не-OpenMP потоков Go он 0). Цена - size копий модели в памяти.
type Pool struct {
	handles chan *Booster
	all     []*Booster
}

// NewPool загружает size независимых хэндлов из одного файла модели. Файл
// читается один раз: все хэндлы гарантированно из одних байт.
func NewPool(path string, size int) (*Pool, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return NewPoolFromBytes(data, size)
}

// NewPoolFromBytes загружает size независимых хэндлов из содержимого файла
// модели (см. LoadBoosterFromBytes).
func NewPoolFromBytes(data []byte, size int) (*Pool, error) {
	if size < 1 {
		return nil, fmt.Errorf("lgbm: pool size must be >= 1, got %d", size)
	}
	p := &Pool{handles: make(chan *Booster, size), all: make([]*Booster, 0, size)}
	for i := range size {
		b, err := LoadBoosterFromBytes(data)
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

// PredictRaw берёт свободный хэндл из пула и считает raw margin.
func (p *Pool) PredictRaw(row []float64) (float64, error) {
	b := <-p.handles
	defer func() { p.handles <- b }()
	return b.PredictRaw(row)
}

// PredictContrib берёт свободный хэндл из пула и считает нативные SHAP contributions.
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
