// The Model families page (site/src/pages/model-families.astro) ties its content to the repository by ids. These
// checks keep it from drifting: every model has notes, every idea names a real model and family, every reference
// key resolves, and every weight count taken from model.yaml still appears in that model's network line.
// Run: npm test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import yaml from 'js-yaml';
import { MODEL_NOTES, FAMILIES, CELLS, COLS, TRAITS, LENSES, BACKLOG_FIT, REFS, SHAP_CASES, SHAP_FEATURES } from '../src/lib/model-families.ts';
import { GLYPHS } from '../src/lib/model-families-glyphs.ts';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
type Card = { status: string; network?: string };
const cards: Record<string, Card> = Object.fromEntries(readdirSync(join(root, 'models'))
  .filter((id) => existsSync(join(root, 'models', id, 'model.yaml')))
  .map((id): [string, Card] => [id, yaml.load(readFileSync(join(root, 'models', id, 'model.yaml'), 'utf-8')) as Card])
  .filter(([, c]) => c.status !== 'in-progress' && c.status !== 'idea'));
const fam = new Set(FAMILIES.map((f) => f.id));

test('model families: every released model has notes, and every note names a real model', () => {
  for (const id of Object.keys(cards)) assert.ok(MODEL_NOTES[id], `no MODEL_NOTES entry for ${id}`);
  for (const id of Object.keys(MODEL_NOTES)) assert.ok(cards[id], `MODEL_NOTES names unknown model ${id}`);
  const traits = new Set(TRAITS.map((t) => t.id));
  for (const [id, n] of Object.entries(MODEL_NOTES)) for (const t of n.t) assert.ok(traits.has(t), `${id}: unknown trait ${t}`);
});

test('model families: ideas name real models and mapped families, once each', () => {
  const seen = new Set<string>();
  for (const c of CELLS) {
    assert.ok(cards[c.m], `idea for unknown model ${c.m}`);
    assert.ok((COLS as readonly string[]).includes(c.f), `${c.m}: family ${c.f} is not a map column`);
    const k = `${c.m}:${c.f}`;
    assert.ok(!seen.has(k), `duplicate idea ${k}`); seen.add(k);
    assert.ok(c.why && c.gap && c.judge && c.change, `${k}: empty field`);
  }
});

test('model families: families, glyphs, references and backlog fits resolve', () => {
  for (const f of FAMILIES) {
    assert.ok(GLYPHS[f.id], `no glyph for ${f.id}`);
    for (const r of f.refs) assert.ok(REFS[r], `${f.id}: unknown reference ${r}`);
    for (const k of [f.np, f.web, f.data]) assert.ok(k >= 1 && k <= 3, `${f.id}: meter out of range`);
  }
  for (const c of COLS) assert.ok(fam.has(c), `map column ${c} is not a family`);
  const backlog = (yaml.load(readFileSync(join(root, 'docs', 'backlog.yaml'), 'utf-8')) as { ideas: { id: string }[] }).ideas.map((i) => i.id);
  for (const [id, b] of Object.entries(BACKLOG_FIT)) {
    assert.ok(backlog.includes(id), `backlog fit for unknown idea ${id}`);
    for (const f of b.f) assert.ok(fam.has(f), `${id}: unknown family ${f}`);
  }
  assert.equal(LENSES.length, 8);
  for (const c of SHAP_CASES) assert.ok(c.twin.length === SHAP_FEATURES.length && c.real.length === SHAP_FEATURES.length, `${c.name}: wrong length`);
});

test('model families: weight counts taken from model.yaml still match its network line', () => {
  const show = (n: number) => [n.toLocaleString('en-US'), `${(n / 1e6).toFixed(2)} M`, `${+(n / 1e6).toFixed(2)} M`, `${+(n / 1e3).toFixed(0)}k`];
  for (const [id, n] of Object.entries(MODEL_NOTES)) {
    if (!n.params || !n.paramsFrom?.startsWith('model.yaml')) continue;
    const net = cards[id].network ?? '';
    const parts = n.paramsFrom.includes(':') ? n.paramsFrom.split(':')[1].split('+').map((s) => s.trim()) : null;
    if (parts) { for (const p of parts) assert.ok(net.includes(p), `${id}: "${p}" not in network line`); continue; }
    assert.ok(show(n.params).some((s) => net.includes(s)), `${id}: ${n.params} not found in network line "${net.slice(0, 120)}…"`);
  }
});
