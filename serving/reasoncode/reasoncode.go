// Пакет reasoncode превращает SHAP contributions строки в ранжированные коды
// причин решения. Ранжирование - топ-K признаков по абсолютному contribution - это
// артефакт объяснимости, чью устойчивость проверяет harness паритета
// (lgbm.TestParityContrib) и который пайплайн decline->explain выдаёт по каждой
// отклонённой транзакции (см. docs/DESIGN.md).
//
// Намеренно чистый Go (без cgo): работает со срезами contributions, уже посчитанными
// нативным предиктором.
package reasoncode

import (
	"math"
	"sort"
)

// TopK возвращает индексы k contributions с наибольшим абсолютным значением, важнейший
// первым. Ничьи разрешаются меньшим индексом ради детерминизма. k зажимается в
// [0, len(contrib)].
//
// Передавайте только contributions признаков - без хвостового base value, которое
// LightGBM добавляет в строку SHAP.
func TopK(contrib []float64, k int) []int {
	n := len(contrib)
	if k < 0 {
		k = 0
	}
	if k > n {
		k = n
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool {
		ai, bi := math.Abs(contrib[idx[a]]), math.Abs(contrib[idx[b]])
		if ai != bi {
			return ai > bi
		}
		return idx[a] < idx[b]
	})
	return idx[:k]
}
