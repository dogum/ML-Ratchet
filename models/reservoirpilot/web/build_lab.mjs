// Bundle the site's ReservoirPilot instrument into one offline HTML file (models/reservoirpilot/lab/index.html):
// the TypeScript instrument, its data packs, the fonts, and tables built from model.yaml, ratchet.yaml and
// training/exam_results.json. Run from the repository root after web/export_instrument.py:
//   node models/reservoirpilot/web/build_lab.mjs
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const model = fileURLToPath(new URL('../', import.meta.url));
const site = join(model, '../../site');
const require = createRequire(join(site, 'package.json'));
const { build } = require('esbuild');
const yaml = require('js-yaml');
const text = path => readFile(path, 'utf8');
const packDir = join(site, 'public/instruments/reservoirpilot');
const output = await build({ entryPoints: [join(model, 'lab/src/entry.ts')], bundle: true,
  format: 'iife', minify: true, write: false, target: 'es2022' });

const packs = [];
for (const name of ['weights', 'dams']) {
  const manifest = JSON.parse(await text(join(packDir, name + '.json')));
  const bytes = (await readFile(join(packDir, name + '.bin'))).toString('base64');
  const safe = JSON.stringify({ manifest, bytes }).replaceAll('<', '\\u003c');
  packs.push('<script type="application/json" id="' + name + '">' + safe + '</script>');
}

const fonts = [];
for (const [pkg, file, family, weight] of [
  ['@fontsource/instrument-serif', 'instrument-serif-latin-400-normal', 'Instrument Serif', '400'],
  ['@fontsource-variable/schibsted-grotesk', 'schibsted-grotesk-latin-wght-normal', 'Schibsted Grotesk Variable', '400 900'],
  ['@fontsource-variable/martian-mono', 'martian-mono-latin-standard-normal', 'Martian Mono Variable', '100 800'],
]) {
  const base = join(site, 'node_modules', pkg);
  const data = (await readFile(join(base, 'files', file + '.woff2'))).toString('base64');
  fonts.push('/* ' + (await text(join(base, 'LICENSE'))).replaceAll('*/', '* /') + ' */' +
    '@font-face{font-family:"' + family + '";font-style:normal;font-weight:' + weight +
    ';font-display:swap;src:url(data:font/woff2;base64,' + data + ') format("woff2")}');
}

const esc = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const tr = (cells, cls = '') => '<tr' + (cls ? ' class="' + cls + '"' : '') + '>' + cells.map(v => '<td>' + v + '</td>').join('') + '</tr>';
const num = v => (typeof v === 'number' ? (Math.abs(v) >= 1 ? v.toFixed(1) : Math.abs(v) >= 0.1 ? v.toFixed(3) : v === 0 ? '0' : v.toPrecision(3)) : esc(v));

// 1. verdict rows, straight from model.yaml
const card = yaml.load(await text(join(model, 'model.yaml')));
const verdictRows = card.exam.results.map(r => {
  const best = r.baselines.map(b => esc(b.name) + ' <b>' + num(b.value) + '</b>').join('<br>');
  return '<tr><td>' + esc(r.metric) + '</td><td><b>' + num(r.model) + '</b></td><td class="bl">' + best + '</td><td class="v ' + r.verdict + '">' + r.verdict +
    '</td></tr><tr class="note"><td colspan="4">' + esc(r.note ?? '') + '</td></tr>';
});

// 2. every component, from the shipped exam log
const ex = JSON.parse(await text(join(model, 'training/exam_results.json')));
const names = { pilot: '<b>ReservoirPilot</b>', tuned_rule_g4: 'Tuned rule (sim g4)', tuned_rule_g1: 'Tuned rule (sim g1)', sop: 'SOP',
  rule: 'WaterData rule curve', historical: 'Historical operations', dp: 'Perfect-foresight DP (bound)' };
const pctf = v => (v * 100).toFixed(1) + '%';
const sgn = v => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(2);
const compRows = (split) => Object.keys(names).map(p => {
  const c = ex[split][p];
  return tr([names[p], c.J.toPrecision(3), c.J_short.toPrecision(2), c.J_flood === 0 ? '0' : c.J_flood.toPrecision(2), pctf(c.shortage_ratio), pctf(c.shortage_days_frac),
    c.flood_days_per_yr.toFixed(1), c.max_excess_ratio.toFixed(2), c.S_mean_frac.toFixed(2), sgn(c.S_end_minus_start_frac), pctf(c.empty_days_frac), pctf(c.full_days_frac)], p === 'pilot' ? 'me' : '');
});

// 3. the ratchet, from ratchet.yaml
const lad = yaml.load(await text(join(model, 'ratchet.yaml'))).ladders[0];
const rungRows = lad.rungs.map(r => tr([esc(r.id), esc(r.change), ...r.metrics.slice(0, 5).map(m => (typeof m === 'number' ? m.toString() : esc(m))),
  '<span class="v ' + (r.decision === 'released' ? 'win' : r.decision === 'kept' ? 'tie' : 'loss') + '">' + r.decision + '</span>', esc(r.why)]));

const global = await text(join(site, 'src/styles/global.css'));
const tokens = global.slice(global.indexOf(':root {'), global.indexOf('*,'));
const replacements = {
  FONTS: fonts.join('\n'), TOKENS: tokens, INSTRUMENT: await text(join(site, 'src/styles/instrument.css')),
  VERDICTS: verdictRows.join('\n'), COMPONENTS: compRows('resopsus_exam').join('\n'), COMPONENTS_MB: compRows('resopsus_exam_mb_ok').join('\n'),
  COMPONENTS_CDEC: compRows('cdec_recent').join('\n'), RUNGS: rungRows.join('\n'), LADDER_NOTE: esc(lad.note),
  PACKS: packs.join('\n'), JS: output.outputFiles[0].text,
};
let html = await text(join(model, 'lab/src/page.html'));
for (const [key, value] of Object.entries(replacements)) html = html.replace('@@' + key + '@@', () => value);
if (html.match(/@@[A-Z0-9_]+@@/)) throw new Error('Unfilled lab placeholder');
await writeFile(join(model, 'lab/index.html'), html);
console.log('Built offline ReservoirPilot lab (' + Buffer.byteLength(html) + ' bytes)');
