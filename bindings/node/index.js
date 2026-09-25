'use strict';
// MahaBodi for Node.js. Methods return plain objects (JSON). Sync methods run on the calling
// thread; the *Async variants run on the libuv pool and return Promises (use them for
// decide/predict so model inference never blocks the event loop).
const path = require('path');
const native = require(path.join(__dirname, `mahabodi.${process.platform}-${process.arch}.node`));

class Bodi {
  constructor(config) {
    this._e = new native.Engine(config === undefined ? undefined : (typeof config === 'string' ? config : JSON.stringify(config)));
  }
  call(method, args = {}) { return JSON.parse(this._e.callJson(method, JSON.stringify(args))); }
  async callAsync(method, args = {}) { return JSON.parse(await this._e.callJsonAsync(method, JSON.stringify(args))); }

  ingest(text, { format = 'auto', source = 'doc' } = {}) { return this.call('ingest', { text, format, source }); }
  ingestBatch(docs) { return this.call('ingest_batch', { docs }); }
  query(q, k = 5) { return this.call('query', { q, k }); }
  traverse(start, hops = 2, limit = 50) { return this.call('traverse', { start, hops, limit }); }
  context(q, k = 5, maxChars = 2000) { return this.call('context', { q, k, max_chars: maxChars }); }
  density() { return this.call('density'); }
  ensureDensity() { return this.call('ensure_density'); }
  fastmemorySearch(q) { return this.call('fastmemory_search', { q }); }
  snapshot() { return this.call('snapshot'); }
  restore(snapshot) { return this.call('restore', { snapshot }); }
  clear() { this.call('clear'); }
  stats() { return this.call('stats'); }

  loadLaya(dir, options = {}) { this.call('load_laya', { dir, options }); }
  loadEmbedder(dir, options = {}) { this.call('load_embedder', { dir, options }); }
  embedText(texts) { return this.callAsync('embed_text', { texts }); }
  // Experience memory: labelled past cases; labels[i] maps question id -> gold label for states[i].
  learn(states, questions, labels, calibrate = 0) { return this.callAsync('learn', { states, questions, labels, calibrate }); }
  decideDefaults() { return this.call('decide_defaults', {}); }
  forget() { this.call('forget'); }
  predict(state, questions) { return this.callAsync('predict', { state, questions }); }
  decide(state, questions, options = null) { return this.callAsync('decide', { state, questions, options }); }
  decideBatch(states, questions, options = null) { return this.callAsync('decide_batch', { states, questions, options }); }
  decideWithMemory(state, questions, { query, k = 3, maxChars = 1200, options = null } = {}) {
    const args = { state, questions, k, max_chars: maxChars, options };
    if (query !== undefined) args.query = query;
    return this.callAsync('decide_with_memory', args);
  }
}

module.exports = { Bodi, version: native.version };
