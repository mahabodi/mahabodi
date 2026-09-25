'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const { Bodi, version } = require('..');

const fixture = (n) => fs.readFileSync(path.join(__dirname, '..', '..', '..', 'crates', 'mahabodi-core', 'tests', 'fixtures', `${n}.md`), 'utf8');

test('ingest, density, query cascade', () => {
  const b = new Bodi();
  const r = b.ingest(fixture('robotics'), { source: 'robotics' });
  assert.strictEqual(r.density.passes, true);
  const q = b.query('spacecraft', 3);
  assert.strictEqual(q.matched, true);
  assert.strictEqual(q.stage, 'exact');
  assert.ok(q.hits.every((h) => h.id.startsWith('F_')));
  const miss = b.query('zzqxv wkkpj');
  assert.strictEqual(miss.matched, false);
  assert.strictEqual(miss.handoff, true);
  assert.strictEqual(b.context('zzqxv wkkpj').context, '');
  assert.ok(version());
});

test('decide defaults are reported', () => {
  const d = new Bodi().decideDefaults();
  assert.strictEqual(d.experience_override_agree, 6);
  assert.strictEqual(d.experience_memory_first_margin, 0.2);
  assert.strictEqual(d.oos_min_similarity, null); // out-of-scope gate off by default
});

test('snapshot/restore and errors', async () => {
  const b = new Bodi({ auto_density: true });
  b.ingest('Refunds take five days. Escalate after two days.', { format: 'text' });
  const b2 = new Bodi();
  b2.restore(b.snapshot());
  assert.deepStrictEqual(b2.query('refunds').hits, b.query('refunds').hits);
  assert.throws(() => b.call('nope'));
  await assert.rejects(b.decide('x', { q: { type: 'noul', instructions: '?' } }));
});

test('laya decisions', { skip: process.env.BODI_LAYA_DIR ? false : 'SKIPPED: set BODI_LAYA_DIR' }, async () => {
  const b = new Bodi();
  b.loadLaya(process.env.BODI_LAYA_DIR);
  const q = { topic: { type: 'choice', instructions: 'What is the topic of `article`?', criteria: { world: 'world news', sports: 'sports', business: 'business', sci_tech: 'science and technology' } } };
  const [a, p] = await Promise.all([
    b.decide({ article: 'The Lakers beat the Celtics 110-102 in overtime on Sunday.' }, q),
    b.predict({ article: 'The Lakers beat the Celtics 110-102 in overtime on Sunday.' }, q),
  ]);
  assert.strictEqual(a.answers.topic.choice, 'sports');
  assert.strictEqual(p.answers.topic.choice, 'sports');
});
