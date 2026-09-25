package mahabodi

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func fixture(t *testing.T, n string) string {
	b, err := os.ReadFile(filepath.Join("..", "..", "..", "crates", "mahabodi-core", "tests", "fixtures", n+".md"))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestIngestQueryDensity(t *testing.T) {
	e, err := New(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	r, err := e.Ingest(fixture(t, "robotics"), "auto", "robotics")
	if err != nil {
		t.Fatal(err)
	}
	if r["density"].(map[string]any)["passes"] != true {
		t.Fatalf("density: %v", r)
	}
	q, err := e.Query("spacecraft", 3)
	if err != nil || !q.Matched || q.Stage != "exact" || len(q.Hits) == 0 {
		t.Fatalf("query: %+v %v", q, err)
	}
	for _, h := range q.Hits {
		if !strings.HasPrefix(h.ID, "F_") {
			t.Fatalf("non-ATF hit %s", h.ID)
		}
	}
	miss, _ := e.Query("zzqxv wkkpj", 5)
	if miss.Matched || !miss.Handoff {
		t.Fatalf("miss: %+v", miss)
	}
	c, _ := e.Context("zzqxv wkkpj", 5, 2000)
	if c["context"] != "" {
		t.Fatalf("context leaked: %v", c)
	}
	if Version() == "" {
		t.Fatal("no version")
	}
}

func TestDecideDefaults(t *testing.T) {
	e, err := New(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	d, err := e.DecideDefaults()
	if err != nil {
		t.Fatal(err)
	}
	if d["experience_override_agree"] != float64(6) || d["experience_memory_first_margin"] != 0.2 || d["oos_min_similarity"] != nil {
		t.Fatalf("defaults %v", d)
	}
}

func TestErrorsSnapshotConcurrency(t *testing.T) {
	if _, err := New(map[string]any{"auto_density": 5}); err == nil {
		t.Fatal("bad config accepted")
	}
	e, _ := New(map[string]any{"auto_density": true})
	var be *Error
	if err := e.Call("nope", nil, nil); !errors.As(err, &be) {
		t.Fatalf("want *Error, got %v", err)
	}
	if _, err := e.Decide("x", map[string]any{"q": map[string]any{"type": "noul", "instructions": "?"}}, nil); err == nil {
		t.Fatal("decide without model must error")
	}
	e.Ingest(fixture(t, "world_events"), "auto", "we")
	snap, _ := e.Snapshot()
	e2, _ := New(nil)
	defer e2.Close()
	if err := e2.Restore(snap); err != nil {
		t.Fatal(err)
	}
	a, _ := e.Query("elevator", 3)
	b, _ := e2.Query("elevator", 3)
	if len(a.Hits) != len(b.Hits) || a.Hits[0].ID != b.Hits[0].ID {
		t.Fatalf("restore mismatch")
	}
	var wg sync.WaitGroup
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if i%8 == 0 {
				if _, err := e.Ingest("Goroutine note on space elevators.", "text", "g"); err != nil {
					t.Error(err)
				}
			} else if _, err := e.Query("elevator", 3); err != nil {
				t.Error(err)
			}
		}(i)
	}
	wg.Wait()
	e.Close()
	e.Close()
	if _, err := e.Query("x", 1); !errors.Is(err, ErrClosed) {
		t.Fatalf("want ErrClosed, got %v", err)
	}
}

func TestLayaDecision(t *testing.T) {
	dir := os.Getenv("BODI_LAYA_DIR")
	if dir == "" {
		t.Skip("SKIPPED: set BODI_LAYA_DIR")
	}
	e, _ := New(nil)
	defer e.Close()
	if err := e.LoadLaya(dir); err != nil {
		t.Fatal(err)
	}
	q := map[string]any{"topic": map[string]any{"type": "choice", "instructions": "What is the topic of `article`?",
		"criteria": map[string]any{"sports": "sports", "world": "world news"}}}
	r, err := e.Decide(map[string]any{"article": "The Lakers beat the Celtics 110-102 in overtime on Sunday."}, q, nil)
	if err != nil {
		t.Fatal(err)
	}
	if r["answers"].(map[string]any)["topic"].(map[string]any)["choice"] != "sports" {
		t.Fatalf("got %v", r)
	}
}
