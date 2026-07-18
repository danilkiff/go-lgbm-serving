// Пакет lgbm - тонкая cgo-обёртка над C API LightGBM для инференса модели,
// обученной в Python. Отдаёт raw margin и нативные SHAP contributions
// (C_API_PREDICT_CONTRIB): коды причин - из того же нативного предиктора, а не из
// повторной реализации.
//
// Прототипы C-ABI объявлены вручную, без #include <LightGBM/c_api.h>: этот
// заголовок тянет C++/Arrow, которые преамбула cgo не компилирует. C API -
// стабильная поверхность extern "C"; прототипы ниже отвечают LightGBM 4.x.
// Версию либы фиксируем соответственно (см. training/pyproject.toml).
//
// Потокобезопасность: предсказание на одном хэндле Booster сериализуется внутри
// C API (LightGBM #3751/#3771). Не делите один Booster между горутинами ради
// пропускной способности - используйте Pool.
package lgbm

/*
#include <stdint.h>
#include <stdlib.h>

typedef void* BoosterHandle;

extern const char* LGBM_GetLastError(void);
extern int LGBM_BoosterLoadModelFromString(const char* model_str, int* out_num_iterations, BoosterHandle* out);
extern int LGBM_BoosterFree(BoosterHandle handle);
extern int LGBM_BoosterGetNumFeature(BoosterHandle handle, int* out_len);
extern int LGBM_BoosterCalcNumPredict(BoosterHandle handle, int num_row, int predict_type, int start_iteration, int num_iteration, int64_t* out_len);
extern int LGBM_BoosterPredictForMat(BoosterHandle handle, const void* data, int data_type, int32_t nrow, int32_t ncol, int is_row_major, int predict_type, int start_iteration, int num_iteration, const char* parameter, int64_t* out_len, double* out_result);

// Флаги линковки заданы здесь (в #cgo), а не в переменной CGO_LDFLAGS: go
// применяет CGO_LDFLAGS дважды (в cgo-объект и повторно на внешней линковке), и
// каждый -rpath дублируется - линковщик это предупреждает. Директива #cgo
// применяется один раз. ${SRCDIR} - каталог этого пакета, поэтому путь к
// lib_lightgbm из uv-venv (Python зафиксирован на 3.12 через .python-version)
// стабилен, и голый `go build ./...` собирается без настройки окружения.
#cgo LDFLAGS: -L${SRCDIR}/../../training/.venv/lib/python3.12/site-packages/lightgbm/lib -Wl,-rpath,${SRCDIR}/../../training/.venv/lib/python3.12/site-packages/lightgbm/lib -l_lightgbm
// rpath на оба префикса Homebrew: /opt/homebrew (Apple Silicon) и /usr/local
// (Intel); несуществующий путь линковщик молча пропускает.
#cgo darwin LDFLAGS: -Wl,-rpath,/opt/homebrew/opt/libomp/lib -Wl,-rpath,/usr/local/opt/libomp/lib
*/
import "C"

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"unsafe"
)

// ErrFeatureCount - вход неверной ширины. Ошибка вызывающего (для HTTP это 4xx),
// в отличие от сбоя самого нативного предиктора; различается через errors.Is.
var ErrFeatureCount = errors.New("feature count mismatch")

// Константы C API (из c_api.h; стабильны в пределах 4.x).
const (
	cDtypeFloat64   = 1
	cPredictRaw     = 1 // C_API_PREDICT_RAW_SCORE - raw margin до сигмоиды
	cPredictContrib = 3 // C_API_PREDICT_CONTRIB - значения SHAP
)

// cPredictParam фиксирует число нативных потоков в 1 на время предсказания.
// Порядок редукции float в многопоточном OpenMP - задокументированный источник
// недетерминизма между запусками (см. README, "Численный паритет"); параллелизм
// держим на уровне Go через Pool. Аллоцируется один раз на весь процесс, чтобы
// горячий путь не делал аллокаций.
var cPredictParam = C.CString("num_threads=1")

// Booster - загруженная модель LightGBM.
//
// Небезопасен для конкурентных вызовов Predict* на одном значении. Для
// параллельного инференса используйте Pool.
type Booster struct {
	handle   C.BoosterHandle
	nFeature int
}

func lastErr() error {
	return fmt.Errorf("lightgbm: %s", C.GoString(C.LGBM_GetLastError()))
}

// LoadBooster загружает модель, ранее записанную Python-методом Booster.save_model.
func LoadBooster(path string) (*Booster, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return LoadBoosterFromBytes(data)
}

// LoadBoosterFromBytes загружает модель из содержимого файла модели. Байты -
// единица идентичности: вызывающий хеширует ровно то, что загружено, без гонки
// с файлом на диске между хешированием и загрузкой.
func LoadBoosterFromBytes(data []byte) (*Booster, error) {
	cstr := C.CString(string(data))
	defer C.free(unsafe.Pointer(cstr))

	var nIter C.int
	var h C.BoosterHandle
	if C.LGBM_BoosterLoadModelFromString(cstr, &nIter, &h) != 0 {
		return nil, lastErr()
	}
	var nf C.int
	if C.LGBM_BoosterGetNumFeature(h, &nf) != 0 {
		C.LGBM_BoosterFree(h)
		return nil, lastErr()
	}
	// PredictRaw отдаёт ровно один margin (out[0]): мультиклассовая модель молча
	// теряла бы остальные классы - отказываем на загрузке, а не в проде. Гейт
	// заодно фиксирует длины вывода: raw - 1, contrib - NumFeature()+1.
	var nOut C.int64_t
	if C.LGBM_BoosterCalcNumPredict(h, 1, C.int(cPredictRaw), 0, -1, &nOut) != 0 {
		C.LGBM_BoosterFree(h)
		return nil, lastErr()
	}
	if nOut != 1 {
		C.LGBM_BoosterFree(h)
		return nil, fmt.Errorf("lgbm: model outputs %d values per row, expected 1 (binary or regression)", int(nOut))
	}
	return &Booster{handle: h, nFeature: int(nf)}, nil
}

// NumFeature возвращает число входных признаков модели.
func (b *Booster) NumFeature() int { return b.nFeature }

// Close освобождает нативную модель. Повторный вызов безопасен.
func (b *Booster) Close() {
	if b.handle != nil {
		C.LGBM_BoosterFree(b.handle)
		b.handle = nil
	}
}

// predictInto прогоняет одну строку через PredictForMat для заданного типа
// предсказания; длину вывода знает вызывающий (горячий путь - один вызов cgo).
func (b *Booster) predictInto(row []float64, predictType, outLen int) ([]float64, error) {
	if len(row) != b.nFeature {
		return nil, fmt.Errorf("lgbm: %w: expected %d features, got %d", ErrFeatureCount, b.nFeature, len(row))
	}
	out := make([]float64, outLen)
	var written C.int64_t
	ret := C.LGBM_BoosterPredictForMat(
		b.handle,
		unsafe.Pointer(&row[0]),
		C.int(cDtypeFloat64),
		C.int32_t(1),
		C.int32_t(b.nFeature),
		C.int(1), // is_row_major
		C.int(predictType),
		C.int(0),  // start_iteration
		C.int(-1), // num_iteration: все
		cPredictParam,
		&written,
		(*C.double)(unsafe.Pointer(&out[0])),
	)
	runtime.KeepAlive(row)
	runtime.KeepAlive(out)
	if ret != 0 {
		return nil, lastErr()
	}
	return out[:int(written)], nil
}

// PredictRaw возвращает raw margin (до сигмоиды) для одной строки - прямой
// аналог Python predict(raw_score=True).
func (b *Booster) PredictRaw(row []float64) (float64, error) {
	out, err := b.predictInto(row, cPredictRaw, 1)
	if err != nil {
		return 0, err
	}
	return out[0], nil
}

// PredictContrib возвращает нативные SHAP contributions для одной строки, длина
// NumFeature()+1 (один выход на строку гарантирован при загрузке). Последний
// элемент - base value; сумма всех элементов равна raw margin (инвариант
// согласованности, который мы проверяем).
func (b *Booster) PredictContrib(row []float64) ([]float64, error) {
	return b.predictInto(row, cPredictContrib, b.nFeature+1)
}
