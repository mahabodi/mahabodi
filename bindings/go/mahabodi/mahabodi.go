// Package mahabodi is the Go binding for Bodi, a System-1 engine for AI agents
// (fastmemory topology memory + Laya decisions), over the mahabodi-ffi C ABI.
//
// Build the native library first (cargo build --release -p mahabodi-ffi); cgo links
// target/release/libmahabodi. Set CGO_LDFLAGS/CGO_CFLAGS to use another location.
//
// An Engine is safe for concurrent use. Call Close to free native memory.
package mahabodi

/*
#cgo CFLAGS: -I${SRCDIR}/../../../crates/mahabodi-ffi/include
#cgo darwin LDFLAGS: -L${SRCDIR}/../../../target/release -lmahabodi -Wl,-rpath,${SRCDIR}/../../../target/release
#cgo linux LDFLAGS: -L${SRCDIR}/../../../target/release -lmahabodi -Wl,-rpath,${SRCDIR}/../../../target/release
#include <stdlib.h>
#include "mahabodi.h"
*/
import "C"

import (
	"encoding/json"
	"errors"
	"runtime"
	"sync"
	"unsafe"
)

// Engine wraps one native Bodi instance.
type Engine struct {
	mu sync.RWMutex
	h  *C.BodiHandle
}

// Error is returned for any failure reported by Bodi.
type Error struct{ Msg string }

func (e *Error) Error() string { return "bodi: " + e.Msg }

// ErrClosed is returned after Close.
var ErrClosed = errors.New("bodi: engine is closed")

// New creates an engine from a config (nil for defaults), e.g. map[string]any{"auto_density": true}.
func New(config any) (*Engine, error) {
	var cfg *C.char
	if config != nil {
		b, err := json.Marshal(config)
		if err != nil {
			return nil, err
		}
		cfg = C.CString(string(b))
		defer C.free(unsafe.Pointer(cfg))
	}
	var cerr *C.char
	h := C.bodi_new(cfg, &cerr)
	if h == nil {
		msg := "bodi_new failed"
		if cerr != nil {
			msg = C.GoString(cerr)
			C.bodi_string_free(cerr)
		}
		return nil, &Error{msg}
	}
	e := &Engine{h: h}
	runtime.SetFinalizer(e, func(e *Engine) { e.Close() })
	return e, nil
}

// Version of the native library.
func Version() string { return C.GoString(C.bodi_version()) }

// CallRaw calls a method with JSON args and returns the raw JSON result.
func (e *Engine) CallRaw(method string, args any) (json.RawMessage, error) {
	if args == nil {
		args = map[string]any{}
	}
	ab, err := json.Marshal(args)
	if err != nil {
		return nil, err
	}
	cm := C.CString(method)
	ca := C.CString(string(ab))
	defer C.free(unsafe.Pointer(cm))
	defer C.free(unsafe.Pointer(ca))
	e.mu.RLock()
	if e.h == nil {
		e.mu.RUnlock()
		return nil, ErrClosed
	}
	out := C.bodi_call(e.h, cm, ca)
	e.mu.RUnlock()
	s := C.GoString(out)
	C.bodi_string_free(out)
	var env struct {
		Ok    json.RawMessage `json:"ok"`
		Error *string         `json:"error"`
	}
	if err := json.Unmarshal([]byte(s), &env); err != nil {
		return nil, err
	}
	if env.Error != nil {
		return nil, &Error{*env.Error}
	}
	return env.Ok, nil
}

// Call calls a method and decodes the result into out (a pointer), or into a map when out is nil.
func (e *Engine) Call(method string, args any, out any) error {
	raw, err := e.CallRaw(method, args)
	if err != nil || out == nil {
		return err
	}
	return json.Unmarshal(raw, out)
}

// Hit is one retrieved memory.
type Hit struct {
	ID    string  `json:"id"`
	Label string  `json:"label"`
	Level string  `json:"level"`
	Block string  `json:"block"`
	Score float64 `json:"score"`
	Text  string  `json:"text,omitempty"`
}

// QueryResult reports how (and whether) a query matched.
type QueryResult struct {
	Query        string  `json:"query"`
	Matched      bool    `json:"matched"`
	Handoff      bool    `json:"handoff"`
	Stage        string  `json:"stage"`
	Confidence   float64 `json:"confidence"`
	TermCoverage float64 `json:"term_coverage"`
	Hits         []Hit   `json:"hits"`
}

func (e *Engine) Ingest(text, format, source string) (map[string]any, error) {
	var m map[string]any
	return m, e.Call("ingest", map[string]any{"text": text, "format": format, "source": source}, &m)
}

// IngestBatch loads many documents with one rebuild: docs = [{"text": ..., "format": ..., "source": ...}].
func (e *Engine) IngestBatch(docs []map[string]any) (map[string]any, error) {
	var m map[string]any
	return m, e.Call("ingest_batch", map[string]any{"docs": docs}, &m)
}

func (e *Engine) Query(q string, k int) (*QueryResult, error) {
	var r QueryResult
	return &r, e.Call("query", map[string]any{"q": q, "k": k}, &r)
}

func (e *Engine) Context(q string, k, maxChars int) (map[string]any, error) {
	var m map[string]any
	return m, e.Call("context", map[string]any{"q": q, "k": k, "max_chars": maxChars}, &m)
}

func (e *Engine) Traverse(start string, hops, limit int) (map[string]any, error) {
	var m map[string]any
	return m, e.Call("traverse", map[string]any{"start": start, "hops": hops, "limit": limit}, &m)
}

func (e *Engine) Density() (map[string]any, error) {
	var m map[string]any
	return m, e.Call("density", nil, &m)
}

func (e *Engine) Snapshot() (json.RawMessage, error) { return e.CallRaw("snapshot", nil) }

func (e *Engine) Restore(snapshot json.RawMessage) error {
	return e.Call("restore", map[string]any{"snapshot": snapshot}, nil)
}

func (e *Engine) LoadEmbedder(dir string) error {
	return e.Call("load_embedder", map[string]any{"dir": dir}, nil)
}

// Learn stores labelled cases as experience memory: labels[i] maps question id -> gold label for states[i].
func (e *Engine) Learn(states []any, questions map[string]any, labels []map[string]any) (map[string]any, error) {
	var m map[string]any
	return m, e.Call("learn", map[string]any{"states": states, "questions": questions, "labels": labels}, &m)
}

func (e *Engine) Forget() error { return e.Call("forget", nil, nil) }

func (e *Engine) LoadLaya(dir string) error {
	return e.Call("load_laya", map[string]any{"dir": dir}, nil)
}

// Decide runs System-1 decisions. questions: {"id": {"type": "choice|score|noul", "instructions": ..., "criteria": ...}}.
func (e *Engine) Decide(state any, questions map[string]any, options map[string]any) (map[string]any, error) {
	var m map[string]any
	return m, e.Call("decide", map[string]any{"state": state, "questions": questions, "options": options}, &m)
}

func (e *Engine) Predict(state any, questions map[string]any) (map[string]any, error) {
	var m map[string]any
	return m, e.Call("predict", map[string]any{"state": state, "questions": questions}, &m)
}

// Close frees the native engine. Safe to call more than once.
func (e *Engine) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.h != nil {
		C.bodi_free(e.h)
		e.h = nil
	}
}
