package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
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
	f := &fakeScorer{res: pipeline.ScoreResult{ID: "abc", Margin: 1.5, Decision: pipeline.Decline}}
	h := scoreHandler(f)

	body, _ := json.Marshal(scoreRequest{Features: []float64{1, 2, 3}})
	rec := httptest.NewRecorder()
	h(rec, httptest.NewRequest(http.MethodPost, "/score", bytes.NewReader(body)))

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
		body, _ := json.Marshal(scoreRequest{Features: []float64{1}})
		rec := httptest.NewRecorder()
		h(rec, httptest.NewRequest(http.MethodPost, "/score", bytes.NewReader(body)))
		if rec.Code != tc.want {
			t.Errorf("%s: status=%d, want %d", tc.name, rec.Code, tc.want)
		}
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

// TestScoreHandlerBodyTooLarge: тело сверх лимита обрезается MaxBytesReader и
// не доходит до скоринга.
func TestScoreHandlerBodyTooLarge(t *testing.T) {
	f := &fakeScorer{}
	h := scoreHandler(f)
	body := append([]byte(`{"features":[`), bytes.Repeat([]byte("1,"), 1<<20)...)
	body = append(body, '1', ']', '}')
	rec := httptest.NewRecorder()
	h(rec, httptest.NewRequest(http.MethodPost, "/score", bytes.NewReader(body)))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d, want 400", rec.Code)
	}
	if f.got != nil {
		t.Fatal("oversized request must not reach the scorer")
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

func TestMetricsHandler(t *testing.T) {
	h := metricsHandler(func() metricsResponse {
		return metricsResponse{Scored: 100, Declined: 5, DeclineRate: 0.05, QueueCap: 1024, Explained: 5}
	})
	rec := httptest.NewRecorder()
	h(rec, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d", rec.Code)
	}
	var got metricsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.Scored != 100 || got.Declined != 5 || got.QueueCap != 1024 {
		t.Fatalf("metrics=%+v", got)
	}
}
