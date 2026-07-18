package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/danilkiff/go-lgbm-serving/lgbm"
	"github.com/danilkiff/go-lgbm-serving/pipeline"
)

// fakeScorer позволяет тестировать HTTP-слой без загрузки нативной модели.
type fakeScorer struct {
	res pipeline.ScoreResult
	err error
	got []float64
}

func (f *fakeScorer) Score(row []float64) (pipeline.ScoreResult, error) {
	f.got = row
	return f.res, f.err
}

func TestScoreHandler(t *testing.T) {
	f := &fakeScorer{res: pipeline.ScoreResult{ID: "abc", Margin: 1.5, Decision: pipeline.Decline, ExplainQueued: true}}
	h := scoreHandler(f)

	rec := httptest.NewRecorder()
	h(rec, httptest.NewRequest(http.MethodPost, "/score", strings.NewReader(`{"features":[1,2,3]}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var resp scoreResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.ID != "abc" || resp.Decision != "decline" || resp.Margin != 1.5 {
		t.Fatalf("response=%+v", resp)
	}
	if !resp.ExplainQueued {
		t.Fatal("explain_queued must pass through to the response")
	}
	if len(f.got) != 3 {
		t.Fatalf("scorer received %d features, want 3", len(f.got))
	}
}

// TestScoreHandlerErrorStatus: неверная ширина входа - 422 (ошибка клиента),
// сбой нативного предиктора - 500 (не вина запроса).
func TestScoreHandlerErrorStatus(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want int
	}{
		{"feature count -> 422", fmt.Errorf("score: %w", lgbm.ErrFeatureCount), http.StatusUnprocessableEntity},
		{"native failure -> 500", errors.New("native predictor failed"), http.StatusInternalServerError},
	}
	for _, tc := range cases {
		h := scoreHandler(&fakeScorer{err: tc.err})
		rec := httptest.NewRecorder()
		h(rec, httptest.NewRequest(http.MethodPost, "/score", strings.NewReader(`{"features":[1]}`)))
		if rec.Code != tc.want {
			t.Errorf("%s: status=%d, want %d", tc.name, rec.Code, tc.want)
		}
	}
}

// TestScoreHandlerNullIsMissing: null в features - это NaN (missing-ветки
// деревьев), а не молчаливый ноль: ноль - другое, легитимное значение признака.
func TestScoreHandlerNullIsMissing(t *testing.T) {
	f := &fakeScorer{}
	h := scoreHandler(f)
	rec := httptest.NewRecorder()
	h(rec, httptest.NewRequest(http.MethodPost, "/score", strings.NewReader(`{"features":[null,1.5,null]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if len(f.got) != 3 || !math.IsNaN(f.got[0]) || f.got[1] != 1.5 || !math.IsNaN(f.got[2]) {
		t.Fatalf("scorer got %v, want [NaN 1.5 NaN]", f.got)
	}
}

func TestScoreHandlerBadJSON(t *testing.T) {
	h := scoreHandler(&fakeScorer{})
	rec := httptest.NewRecorder()
	h(rec, httptest.NewRequest(http.MethodPost, "/score", bytes.NewReader([]byte("{not json"))))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d, want 400", rec.Code)
	}
}

// TestScoreHandlerTrailingGarbage: запрос - ровно один JSON-объект; хвост после
// него (мусор или второй объект) - 400 и до скоринга не доходит. Хвост из одних
// пробелов - это конец корректного тела, а не мусор.
func TestScoreHandlerTrailingGarbage(t *testing.T) {
	f := &fakeScorer{}
	h := scoreHandler(f)
	rec := httptest.NewRecorder()
	h(rec, httptest.NewRequest(http.MethodPost, "/score",
		strings.NewReader(`{"features":[1]}{"features":[2]}`)))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("trailing object: status=%d, want 400", rec.Code)
	}
	if f.got != nil {
		t.Fatal("request with trailing data must not reach the scorer")
	}

	rec = httptest.NewRecorder()
	h(rec, httptest.NewRequest(http.MethodPost, "/score", strings.NewReader(`{"features":[1]}`+"\n \t")))
	if rec.Code != http.StatusOK {
		t.Fatalf("trailing whitespace: status=%d body=%s", rec.Code, rec.Body.String())
	}
}

// TestScoreHandlerBodyTooLarge: тело сверх лимита - это 413 (не общий 400) и до
// скоринга не доходит.
func TestScoreHandlerBodyTooLarge(t *testing.T) {
	f := &fakeScorer{}
	h := scoreHandler(f)
	body := append([]byte(`{"features":[`), bytes.Repeat([]byte("1,"), 1<<20)...)
	body = append(body, '1', ']', '}')
	rec := httptest.NewRecorder()
	h(rec, httptest.NewRequest(http.MethodPost, "/score", bytes.NewReader(body)))
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status=%d, want 413", rec.Code)
	}
	if f.got != nil {
		t.Fatal("oversized request must not reach the scorer")
	}
}

// TestModelVersion: версия начинается с пути и меняется вместе с байтами модели -
// это и есть привязка объяснения к конкретной модели.
func TestModelVersion(t *testing.T) {
	v1 := modelVersion("m.txt", []byte("tree v1"))
	if !strings.HasPrefix(v1, "m.txt@") || len(v1) != len("m.txt")+1+16 {
		t.Fatalf("version=%q, want path + '@' + 16 hex chars", v1)
	}
	if modelVersion("m.txt", []byte("tree v1")) != v1 {
		t.Fatal("same bytes must give the same version")
	}
	if modelVersion("m.txt", []byte("tree v2")) == v1 {
		t.Fatal("different bytes must give a different version")
	}
}

type fakeStore struct {
	exp pipeline.Explanation
	ok  bool
}

func (f fakeStore) Get(string) (pipeline.Explanation, bool) { return f.exp, f.ok }

func TestExplainHandlerFound(t *testing.T) {
	exp := pipeline.Explanation{ID: "abc", Margin: 2.0, Reasons: []pipeline.ReasonCode{{Feature: 5, Contribution: 1.1}}}
	h := explainHandler(fakeStore{exp: exp, ok: true})

	req := httptest.NewRequest(http.MethodGet, "/explain/abc", nil)
	req.SetPathValue("id", "abc")
	rec := httptest.NewRecorder()
	h(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var got pipeline.Explanation
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.ID != "abc" || len(got.Reasons) != 1 || got.Reasons[0].Feature != 5 {
		t.Fatalf("explanation=%+v", got)
	}
}

func TestExplainHandlerNotFound(t *testing.T) {
	h := explainHandler(fakeStore{ok: false})
	req := httptest.NewRequest(http.MethodGet, "/explain/missing", nil)
	req.SetPathValue("id", "missing")
	rec := httptest.NewRecorder()
	h(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status=%d, want 404", rec.Code)
	}
}
