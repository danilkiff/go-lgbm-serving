package reasoncode

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
)

// Code - код причины (adverse-action) для признака: стабильный идентификатор и
// человекочитаемая метка.
type Code struct {
	Code  string `json:"code"`
	Label string `json:"label"`
}

// Catalog отображает индексы признаков в коды причин. Признаки без явной записи
// получают обобщённый запасной код по индексу, чтобы у объяснения всегда был код.
type Catalog struct {
	byIndex map[int]Code
}

// NewCatalog строит каталог из отображения индекс признака -> Code. Пустая карта
// (или nil-*Catalog) допустима: тогда любой Lookup вернёт запасной код.
func NewCatalog(byIndex map[int]Code) *Catalog {
	return &Catalog{byIndex: byIndex}
}

// Lookup возвращает код причины для признака, откатываясь к обобщённому
// "R<index>" / "feature <index>", если признака нет (или каталог nil), - так
// отклонение никогда не остаётся без кода.
func (c *Catalog) Lookup(feature int) Code {
	if c != nil {
		if code, ok := c.byIndex[feature]; ok {
			return code
		}
	}
	return Code{Code: fmt.Sprintf("R%d", feature), Label: fmt.Sprintf("feature %d", feature)}
}

// Direction сообщает, толкнул ли contribution оценку в сторону неблагоприятного решения.
// В этой схеме большой raw margin означает больший риск, поэтому
// неотрицательный contribution риск увеличил.
func Direction(contribution float64) string {
	if contribution >= 0 {
		return "increased risk"
	}
	return "decreased risk"
}

// LoadCatalog читает JSON-объект, отображающий индекс признака (строковый ключ) в
// Code, например {"21": {"code": "R21", "label": "transaction amount"}}.
func LoadCatalog(path string) (*Catalog, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw map[string]Code
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, fmt.Errorf("reasoncode: parse %s: %w", path, err)
	}
	byIndex := make(map[int]Code, len(raw))
	for k, v := range raw {
		i, err := strconv.Atoi(k)
		if err != nil {
			return nil, fmt.Errorf("reasoncode: bad feature index %q in %s", k, path)
		}
		byIndex[i] = v
	}
	return NewCatalog(byIndex), nil
}
