'use strict';
// PostgreSQL store through the Node binding (native library built with the `postgres` feature).
// Skipped unless MAHABODI_TEST_PG_DSN is set; skipped at store_open when this build has no store,
// mirroring bindings/python/tests/test_store_pg.py.
const test = require('node:test');
const assert = require('node:assert');
const { Bodi } = require('..');

const DSN = process.env.MAHABODI_TEST_PG_DSN;

const DOCS = [
  { text: 'Travel over $500 must be approved by a manager.', source: 'policy1' },
  { text: 'Expense reports are due within 30 days.', source: 'policy2' },
  { text: 'Refunds above $5,000 need the finance director.', source: 'policy3' },
];

test('store round trip (PostgreSQL)', { skip: DSN ? false : 'SKIPPED: set MAHABODI_TEST_PG_DSN' }, (t) => {
  const b = new Bodi();
  // Unique namespace per run. The engine exposes no store_drop method, so the namespace's rows are
  // left behind (the Rust test cleans up with direct SQL, which the binding cannot do).
  const ns = `tnode_${process.pid}`;
  try {
    b.call('store_open', { dsn: DSN, namespace: ns, vector_type: 'vector', create: true });
  } catch (e) { // a binary built without the postgres feature
    t.skip(`store not available in this build: ${e.message}`);
    return;
  }
  b.call('store_ingest_batch', { docs: DOCS });
  b.call('store_build_index');
  const q = b.call('store_query', { q: 'who approves travel', k: 10, mode: 'hybrid' });
  assert.ok(typeof q.stage === 'string' && q.stage.length > 0, JSON.stringify(q));
  assert.ok(q.hits.length > 0, JSON.stringify(q));
  assert.ok(q.hits[0].text.includes('approved by a manager'), JSON.stringify(q.hits[0]));
  assert.strictEqual(q.handoff, false);
  // No hub fallback in the store: a query nothing matches is an empty handoff.
  const miss = b.call('store_query', { q: 'zzqx unrelated gibberish', k: 5 });
  assert.ok(miss.handoff === true || miss.stage === 'hub' || miss.stage === 'empty_memory', JSON.stringify(miss));
});

// Non-skipping: proves the `postgres` feature is compiled into this build, with no server needed.
// A build without the feature answers "unknown method 'store_open'"; a build with it fails to
// connect to port 1 with a store/postgres error.
test('store feature compiled in (unreachable DSN)', () => {
  const b = new Bodi();
  let msg = '';
  try {
    b.call('store_open', { dsn: 'host=127.0.0.1 port=1 user=x dbname=x connect_timeout=1', namespace: 'tnode_unreach', vector_type: 'vector', create: false });
  } catch (e) { msg = String((e && e.message) || e); }
  assert.ok(msg.length > 0, 'store_open unexpectedly succeeded against port 1');
  assert.ok(!msg.includes('unknown method'), `store not compiled into this build: ${msg}`);
  assert.ok(/postgres|store|connect/i.test(msg), msg);
});
