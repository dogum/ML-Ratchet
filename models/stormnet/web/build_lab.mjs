// Bundle the same instrument used by the site into one offline HTML file.
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const model = fileURLToPath(new URL('../', import.meta.url));
const site = join(model, '../../site');
const require = createRequire(join(site, 'package.json'));
const { build } = require('esbuild');
const text = path => readFile(path, 'utf8');
const packDir = join(site, 'public/instruments/stormnet');
const output = await build({ entryPoints: [join(model, 'lab/src/entry.ts')], bundle: true,
  format: 'iife', minify: true, write: false, target: 'es2022' });

const packs = [];
for (const name of ['weights', 'storms']) {
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

const exam = JSON.parse(await text(join(model, 'training/exam.json')));
const rowLabels = {
  'All eligible hours': 'All eligible hours',
  'Storm-strength minimum': 'Storm strength (≤ −50 nT)',
  'Intense minimum': 'Intense (≤ −100 nT)',
  'Extreme minimum': 'Extreme (≤ −150 nT)',
  'Quiet false alarms': 'Quiet (> −50 nT), false alarms',
};
const tableRow = cells => '<tr>' + cells.map(v => '<td>' + v + '</td>').join('') + '</tr>';
const fixed = v => v.toFixed(3);
const signed = v => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(3);
const rows = [], intervals = [];
for (const [name, r] of Object.entries(exam.rows)) {
  const label = rowLabels[name];
  if (!label) throw new Error('Missing exam row label for ' + name);
  rows.push(tableRow([label, r.issues.toLocaleString('en-US') + ' / ' + r.blocks,
    ...Object.values(r.scores).map(fixed), r.verdict === 'info' ? 'Inconclusive' : r.verdict]));
  const unit = r.unit === '%' ? ' pp' : ' nT';
  intervals.push(tableRow([label, signed(r.difference) + unit,
    r.difference_ci95 ? r.difference_ci95.map(signed).join(' to ') + unit : 'One storm',
    r.coverage_pct.toFixed(1) + '%',
    r.coverage_ci95 ? r.coverage_ci95.map(v => v.toFixed(1)).join(' to ') + '%' : 'One storm']));
}
const reserve = JSON.parse(await text(join(model, 'training/exam_reserve.json')))['v1.0.1'];
const rows2 = [], intervals2 = [];
for (const [name, r] of Object.entries(reserve)) {
  const label = rowLabels[name];
  rows2.push(tableRow([label, r.issues.toLocaleString('en-US') + ' / ' + r.blocks,
    ...Object.values(r.scores).map(fixed), r.verdict === 'info' ? 'Inconclusive' : r.verdict]));
  const unit = r.unit === '%' ? ' pp' : ' nT';
  intervals2.push(tableRow([label, signed(r.difference) + unit, r.difference_ci95.map(signed).join(' to ') + unit,
    r.coverage_pct.toFixed(1) + '%', r.coverage_ci95.map(v => v.toFixed(1)).join(' to ') + '%']));
}
const global = await text(join(site, 'src/styles/global.css'));
const tokens = global.slice(global.indexOf(':root {'), global.indexOf('*,'));
const replacements = {
  FONTS: fonts.join('\n'), TOKENS: tokens, INSTRUMENT: await text(join(site, 'src/styles/instrument.css')),
  ROWS: rows.join('\n'), INTERVALS: intervals.join('\n'), ROWS2: rows2.join('\n'), INTERVALS2: intervals2.join('\n'),
  PACKS: packs.join('\n'),
  TWINS: (await readFile(join(model, 'dev-twins.png'))).toString('base64'), JS: output.outputFiles[0].text,
};
let html = await text(join(model, 'lab/src/page.html'));
for (const [key, value] of Object.entries(replacements)) html = html.replace('@@' + key + '@@', () => value);
if (html.match(/@@[A-Z0-9]+@@/)) throw new Error('Unfilled lab placeholder');
await writeFile(join(model, 'lab/index.html'), html);
console.log('Built offline StormNet lab (' + Buffer.byteLength(html) + ' bytes)');
