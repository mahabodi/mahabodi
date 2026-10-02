package mahabodi

// PostgreSQL store through the Go binding (native library built with the `postgres` feature).
// Skipped unless MAHABODI_TEST_PG_DSN is set; skipped at store_open when this build has no store,
// mirroring bindings/python/tests/test_store_pg.py.

import (
	"fmt"
	"os"
	"strings"
	"testing"
)

func TestStoreRoundTrip(t *testing.T) {
	dsn := os.Getenv("MAHABODI_TEST_PG_DSN")
	if dsn == "" {
		t.Skip("SKIPPED: set MAHABODI_TEST_PG_DSN")
	}
	e, err := New(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	// Unique namespace per run. The engine exposes no store_drop method, so the namespace's rows are
	// left behind (the Rust test cleans up with direct SQL, which the binding cannot do).
	ns := fmt.Sprintf("tgo_%d", os.Getpid())
	if err := e.Call("store_open", map[string]any{"dsn": dsn, "namespace": ns, "vector_type": "vector", "create": true}, nil); err != nil {
		t.Skipf("store not available in this build: %v", err) // a binary built without the postgres feature
	}
	docs := []map[string]any{
		{"text": "Travel over $500 must be approved by a manager.", "source": "policy1"},
		{"text": "Expense reports are due within 30 days.", "source": "policy2"},
		{"text": "Refunds above $5,000 need the finance director.", "source": "policy3"},
	}
	if err := e.Call("store_ingest_batch", map[string]any{"docs": docs}, nil); err != nil {
		t.Fatal(err)
	}
	if err := e.Call("store_build_index", nil, nil); err != nil {
		t.Fatal(err)
	}
	var q QueryResult
	if err := e.Call("store_query", map[string]any{"q": "who approves travel", "k": 10, "mode": "hybrid"}, &q); err != nil {
		t.Fatal(err)
	}
	if q.Stage == "" || len(q.Hits) == 0 {
		t.Fatalf("store_query: %+v", q)
	}
	if !strings.Contains(q.Hits[0].Text, "approved by a manager") {
		t.Fatalf("top hit is not the travel doc: %+v", q.Hits[0])
	}
	if q.Handoff {
		t.Fatalf("handoff on a matching query: %+v", q)
	}
	// No hub fallback in the store: a query nothing matches is an empty handoff.
	var miss QueryResult
	if err := e.Call("store_query", map[string]any{"q": "zzqx unrelated gibberish", "k": 5}, &miss); err != nil {
		t.Fatal(err)
	}
	if !miss.Handoff && miss.Stage != "hub" && miss.Stage != "empty_memory" {
		t.Fatalf("miss: %+v", miss)
	}
}

// Non-skipping: proves the `postgres` feature is compiled into this build, with no server needed.
// A build without the feature answers "unknown method 'store_open'"; a build with it fails to
// connect to port 1 with a store/postgres error.
func TestStoreFeatureCompiledIn(t *testing.T) {
	e, err := New(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	err = e.Call("store_open", map[string]any{"dsn": "host=127.0.0.1 port=1 user=x dbname=x connect_timeout=1", "namespace": "tgo_unreach", "vector_type": "vector", "create": false}, nil)
	if err == nil {
		t.Fatal("store_open unexpectedly succeeded against port 1")
	}
	msg := err.Error()
	if strings.Contains(msg, "unknown method") {
		t.Fatalf("store not compiled into this build: %s", msg)
	}
	if !strings.Contains(strings.ToLower(msg), "postgres") && !strings.Contains(strings.ToLower(msg), "store") && !strings.Contains(strings.ToLower(msg), "connect") {
		t.Fatalf("unexpected error: %s", msg)
	}
}
