// Run: node tests/js/test_voice_trigger.mjs   (no dependencies, no DOM). Mirrors tests/test_voice.py::test_split_trigger_*
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const require = createRequire(import.meta.url);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const V = require(path.join(root, 'static/js/voice.js'));
let n = 0;
const t = (name, fn) => { fn(); n++; console.log('ok -', name); };

t('trigger phrase at the end is stripped', () => {
  const cases = [
    ['Fix the login bug in auth.py go ahead now', 'Fix the login bug in auth.py'],
    ['Fix the login bug. Go ahead, now.', 'Fix the login bug.'],
    ['Fix the login bug, GO AHEAD NOW!', 'Fix the login bug'],
    ['refactor utils.py -- process now', 'refactor utils.py'],
    ['rename it send it now', 'rename it'],
    ['do the thing go ahead now please', 'do the thing'],
    ['isko fix karo ab bhejo', 'isko fix karo'],
    ['isko fix karo, Process Karo.', 'isko fix karo'],
    ['sab theek hai bhej do yaar', 'sab theek hai'],
    ['इसे ठीक करो अब भेजो', 'इसे ठीक करो'],
    ['इसे ठीक करो, प्रोसेस करो।', 'इसे ठीक करो'],
    ['go ahead now go ahead now', ''],
    ['Go ahead now.', ''],
  ];
  for (const [input, want] of cases) assert.deepEqual(V.splitTrigger(input), { text: want, triggered: true }, input);
});

t('a phrase that is not at the end never triggers', () => {
  for (const s of ['go ahead and refactor the parser', 'please process now the queue is empty and then stop', 'fix the bug', 'ahead now', '', '   '])
    assert.deepEqual(V.splitTrigger(s), { text: s.trim(), triggered: false }, s);
});

t('custom phrases, longest first', () => {
  assert.deepEqual(V.splitTrigger('deploy it please engage', 'engage, make it so'), { text: 'deploy it please', triggered: true });
  assert.deepEqual(V.splitTrigger('go ahead now', 'make it so'), { text: 'go ahead now', triggered: false });
  assert.deepEqual(V.parsePhrases('go ahead, go ahead now')[0], ['go', 'ahead', 'now']);
  assert.deepEqual(V.parsePhrases(' , ,'), []);
});

t('encodeWav writes a 16 kHz mono 16-bit wav', () => {
  const sr = 48000, x = new Float32Array(sr / 2).map((_, i) => Math.sin(i / 20) * 0.5);
  const w = V.encodeWav(x, sr, 16000), dv = new DataView(w.buffer);
  assert.equal(String.fromCharCode(...w.slice(0, 4)), 'RIFF');
  assert.equal(String.fromCharCode(...w.slice(8, 12)), 'WAVE');
  assert.equal(dv.getUint16(20, true), 1); assert.equal(dv.getUint16(22, true), 1);
  assert.equal(dv.getUint32(24, true), 16000); assert.equal(dv.getUint16(34, true), 16);
  assert.equal(dv.getUint32(40, true), 8000 * 2); assert.equal(w.length, 44 + 8000 * 2);
  assert.ok(Math.max(...new Int16Array(w.buffer.slice(44))) > 10000);
});

t('speakable drops code and markdown', () => {
  assert.equal(V.speakable('# Title\nUse `foo()` now.\n```py\nprint(1)\n```\nSee [docs](http://x).'), 'Title Use foo() now. (code omitted). See docs.');
});

console.log(`${n} passed`);
