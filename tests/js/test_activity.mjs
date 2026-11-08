// Run: node tests/js/test_activity.mjs   (no dependencies, no DOM)
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const require = createRequire(import.meta.url);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const A = require(path.join(root, 'static/js/activity.js'));
let n = 0;
const t = (name, fn) => { fn(); n++; console.log('ok -', name); };

const ev = (o) => Object.assign({ uid: 'u' + Math.random(), ts: '2026-01-01T00:00:00.000Z', request_id: 'r1', thread_id: 't1', phase: 'RETRIEVE', agent: 'A', action: 'grep', status: 'info', detail: '', data: {} }, o);

t('phaseColor is stable and known phases differ', () => {
  assert.equal(A.phaseColor('TEST'), A.phaseColor('test'));
  assert.notEqual(A.phaseColor('TEST'), A.phaseColor('FIX'));
  assert.match(A.phaseColor('WEIRD'), /^hsl\(\d+ 70% 50%\)$/);
  A.PHASES.forEach(p => assert.ok(Number.isInteger(A.phaseHue(p))));
});

t('eventKind defaults to log', () => {
  assert.equal(A.eventKind({}), 'log');
  assert.equal(A.eventKind({ kind: 'ACTION' }), 'action');
  assert.equal(A.eventKind({ kind: 'job' }), 'job');
  assert.equal(A.eventKind({ kind: 'bogus' }), 'log');
  assert.equal(A.eventKind(null), 'log');
});

t('normalizeJob uppercases state and computes pct', () => {
  const j = A.normalizeJob({ job_id: 'j', state: 'running', progress: '42', total: 100 });
  assert.equal(j.state, 'RUNNING'); assert.equal(j.pct, 42); assert.ok(A.isActiveJob(j));
  assert.equal(A.normalizeJob({ state: 'completed' }).pct, null);
  assert.ok(A.isTerminalJob(A.normalizeJob({ state: 'Failed' })));
  assert.equal(A.normalizeJob({ state: 'nonsense' }).state, 'QUEUED');
  assert.equal(A.normalizeJob(null), null);
});

t('searchModeInfo labels', () => {
  assert.equal(A.searchModeInfo({ vector_status: 'none' }).label, 'Agentic search');
  assert.equal(A.searchModeInfo({ vector_status: 'none' }).canBuild, true);
  assert.equal(A.searchModeInfo({ vector_status: 'ready', retrieval_mode: 'hybrid' }).label, 'Hybrid (vectors ready)');
  assert.equal(A.searchModeInfo({ vector_status: 'failed' }).label, 'Vectorize failed');
  const run = A.searchModeInfo({ vector_status: 'running', job: { state: 'running', progress: 21, total: 50 } });
  assert.equal(run.label, 'Vectorizing 42%'); assert.equal(run.pct, 42); assert.equal(run.canBuild, false);
  assert.equal(A.searchModeInfo({ vector_status: 'none' }, { state: 'RUNNING', progress: 1, total: 4 }).label, 'Vectorizing 25%');
  assert.equal(A.searchModeInfo({ vector_status: 'running' }, { state: 'COMPLETED' }).label, 'Hybrid (vectors ready)');
  assert.equal(A.searchModeInfo(null), null);
});

t('mergeActionEvents pairs start+closing via parent_uid', () => {
  const rows = A.mergeActionEvents([
    ev({ uid: 's1', ts: '2026-01-01T00:00:01.000Z', status: 'start', action: 'run tests', phase: 'TEST' }),
    ev({ uid: 'e1', ts: '2026-01-01T00:00:03.000Z', status: 'ok', parent_uid: 's1', duration_ms: 2000, detail: 'done', data: { n: 3 }, phase: 'TEST', action: 'run tests' }),
    ev({ uid: 's2', ts: '2026-01-01T00:00:04.000Z', status: 'start', action: 'open' }),
    ev({ uid: 'i1', ts: '2026-01-01T00:00:02.000Z', status: 'info', action: 'note' }),
  ]);
  assert.equal(rows.length, 3);
  assert.deepEqual(rows.map(r => r.uid), ['s1', 'i1', 's2']);       // time-ordered
  assert.equal(rows[0].status, 'ok'); assert.equal(rows[0].open, false);
  assert.equal(rows[0].duration_ms, 2000); assert.equal(rows[0].detail, 'done'); assert.deepEqual(rows[0].data, { n: 3 });
  assert.equal(rows[2].open, true); assert.equal(rows[2].status, 'start');
});

t('mergeActionEvents: closing before start, duplicates, orphans, fail', () => {
  const rows = A.mergeActionEvents([
    ev({ uid: 'e1', ts: '2026-01-01T00:00:02.000Z', status: 'fail', parent_uid: 's1', detail: 'boom' }),
    ev({ uid: 's1', ts: '2026-01-01T00:00:01.000Z', status: 'start' }),
    ev({ uid: 's1', ts: '2026-01-01T00:00:01.000Z', status: 'start' }),          // replayed duplicate
    ev({ uid: 'o1', ts: '2026-01-01T00:00:03.000Z', status: 'ok', parent_uid: 'missing' }),
  ]);
  assert.equal(rows.length, 2);
  assert.equal(rows[0].status, 'fail'); assert.equal(rows[0].detail, 'boom');
  assert.equal(rows[1].uid, 'o1'); assert.equal(rows[1].open, false);
  assert.deepEqual(A.mergeActionEvents(null), []);
});

t('groupByRequest groups, counts, status, query', () => {
  const rows = A.mergeActionEvents([
    ev({ uid: 'a', request_id: 'r1', ts: '2026-01-01T00:00:00.000Z', phase: 'INTAKE', status: 'info', data: { query: 'How does   login work?' } }),
    ev({ uid: 'b', request_id: 'r1', ts: '2026-01-01T00:00:01.000Z', status: 'start' }),
    ev({ uid: 'c', request_id: 'r1', ts: '2026-01-01T00:00:02.000Z', status: 'fail', parent_uid: 'b' }),
    ev({ uid: 'd', request_id: 'r2', ts: '2026-01-01T00:05:00.000Z', status: 'start' }),
    ev({ uid: 'x', request_id: null, thread_id: null, ts: '2026-01-01T00:00:00.000Z', status: 'ok' }),
  ]);
  const groups = A.groupByRequest(rows, { queries: { r2: 'second question' }, now: Date.parse('2026-01-01T00:05:10.000Z') });
  assert.equal(groups.length, 3);
  assert.equal(groups[0].key, 'r2');                                  // newest first
  assert.equal(groups[0].status, 'running'); assert.equal(groups[0].query, 'second question');
  assert.equal(groups[0].durationMs, 10000);
  const g1 = groups.find(g => g.key === 'r1');
  assert.equal(g1.status, 'fail'); assert.equal(g1.query, 'How does login work?');
  assert.deepEqual(g1.counts, { total: 2, ok: 1, fail: 1, skip: 0, open: 0 });
  assert.equal(g1.durationMs, 2000);
  assert.equal(groups.find(g => g.key === 'system').label, 'System activity');
});

t('filterRows', () => {
  const rows = A.mergeActionEvents([
    ev({ uid: '1', phase: 'TEST', status: 'ok', agent: 'Runner' }),
    ev({ uid: '2', phase: 'FIX', status: 'fail', agent: 'Patcher', thread_id: 't2', detail: 'Syntax Error' }),
  ]);
  assert.equal(A.filterRows(rows, {}).length, 2);
  assert.equal(A.filterRows(rows, { phases: new Set(['FIX']) }).length, 1);
  assert.equal(A.filterRows(rows, { phases: ['TEST'] }).length, 1);
  assert.equal(A.filterRows(rows, { failuresOnly: true })[0].uid, '2');
  assert.equal(A.filterRows(rows, { text: 'syntax' }).length, 1);
  assert.equal(A.filterRows(rows, { threadId: 't1' })[0].uid, '1');
});

t('selectVisible applies DOM budget to open groups only', () => {
  const mk = (k, n) => ({ key: k, rows: Array.from({ length: n }, (_, i) => ({ i })) });
  const r = A.selectVisible([mk('a', 6), mk('b', 6), mk('c', 6)], g => g.key !== 'b', 8);
  assert.equal(r.shown, 8); assert.equal(r.hidden, 4);
  assert.equal(r.groups[0].rows.length, 6); assert.equal(r.groups[1].open, false);
  assert.equal(r.groups[2].rows.length, 2); assert.equal(r.groups[2].truncated, 4);
  assert.equal(r.groups[2].rows[0].i, 4);                              // newest rows are kept
});

t('fmtDuration', () => {
  assert.equal(A.fmtDuration(420), '420 ms'); assert.equal(A.fmtDuration(1500), '1.5 s');
  assert.equal(A.fmtDuration(12000), '12 s'); assert.equal(A.fmtDuration(125000), '2m 5s');
  assert.equal(A.fmtDuration(null), '');
});

/* Markdown renderer (first block of app.js, DOM-free): Delivery notes callout + XSS safety */
const appSrc = readFileSync(path.join(root, 'static/js/app.js'), 'utf8');
const renderSrc = appSrc.slice(0, appSrc.indexOf('/* ================= State & helpers'));
const Render = new Function(renderSrc + '; return Render;')();

t('Delivery notes render as a callout; body untouched', () => {
  const md = 'Answer **text**.\n\n```py\nx = 1\n```\n\n### Delivery notes\n**Assumptions:**\n- uses sqlite\n**Unverified:**\n- <script>alert(1)</script>';
  const html = Render.renderMarkdown(md);
  assert.ok(html.includes('delivery-notes'));
  assert.ok(html.indexOf('Answer') < html.indexOf('delivery-notes'));
  assert.ok(html.includes('uses sqlite'));
  assert.ok(!html.includes('<script>'));
  assert.ok(html.includes('&lt;script&gt;'));
  assert.equal((html.match(/delivery-notes/g) || []).length >= 1, true);
});

t('Delivery notes marker inside a code fence is ignored', () => {
  const html = Render.renderMarkdown('```\n### Delivery notes\n```\nplain');
  assert.ok(!html.includes('<details'));
});

t('Markdown without notes unchanged in structure', () => {
  const html = Render.renderMarkdown('# Title\n\n- a\n- b');
  assert.ok(!html.includes('delivery-notes')); assert.ok(html.includes('Title'));
});

console.log(`\n${n} tests passed`);
