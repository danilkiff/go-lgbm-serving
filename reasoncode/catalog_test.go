package reasoncode

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCatalogLookup(t *testing.T) {
	c := NewCatalog(map[int]Code{21: {Code: "AMT", Label: "transaction amount"}})
	if got := c.Lookup(21); got.Code != "AMT" || got.Label != "transaction amount" {
		t.Errorf("mapped lookup=%+v", got)
	}
	if got := c.Lookup(7); got.Code != "R7" || got.Label != "feature 7" {
		t.Errorf("fallback lookup=%+v, want generic R7", got)
	}
	var nilCat *Catalog
	if got := nilCat.Lookup(3); got.Code != "R3" {
		t.Errorf("nil-catalog lookup=%+v, want generic R3", got)
	}
}

func TestDirection(t *testing.T) {
	if Direction(1.2) != "increased risk" {
		t.Error("positive contribution should increase risk")
	}
	if Direction(-0.5) != "decreased risk" {
		t.Error("negative contribution should decrease risk")
	}
}

func TestLoadCatalog(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "codes.json")
	body := `{"21":{"code":"AMT","label":"amount"},"4":{"code":"V4","label":"v4"}}`
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	c, err := LoadCatalog(p)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got := c.Lookup(21); got.Code != "AMT" {
		t.Errorf("loaded lookup(21)=%+v", got)
	}
	if got := c.Lookup(4); got.Label != "v4" {
		t.Errorf("loaded lookup(4)=%+v", got)
	}
}

func TestLoadCatalogBadIndex(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(p, []byte(`{"notanint":{"code":"X","label":"x"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCatalog(p); err == nil {
		t.Fatal("expected an error for a non-integer feature key")
	}
}
