// Bundle the same instrument the site uses into one offline HTML file: models/snowsense/lab/index.html.
// Run from the repository root after `python models/snowsense/web/export_instrument.py`:  node models/snowsense/web/build_lab.mjs
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const model = fileURLToPath(new URL('../', import.meta.url));
const site = join(model, '../../site');
const require = createRequire(join(site, 'package.json'));
const { build } = require('esbuild');
const text = path => readFile(path, 'utf8');
const json = async path => JSON.parse(await text(path));
const packDir = join(site, 'public/instruments/snowsense');
const output = await build({ entryPoints: [join(model, 'lab/src/entry.ts')], bundle: true,
  format: 'iife', minify: true, write: false, target: 'es2022' });

const packs = [];
for (const name of ['weights', 'stations']) {
  const manifest = await json(join(packDir, name + '.json'));
  const bytes = (await readFile(join(packDir, name + '.bin'))).toString('base64');
  const safe = JSON.stringify({ manifest, bytes }).replaceAll('<', '\u003c');
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

// Every number below comes from the shipped training JSONs, unchanged.
const S = await json(join(model, 'training/final_summary.json'));
const R = await json(join(model, 'training/ratchet.json'));
const E = await json(join(model, 'training/exam_by_rung.json'));
const RSall = await json(join(model, 'training/rescore_summary.json'));   // re-score, 18 Nov 2026
const RS = RSall.medians;
const row = cells => '<tr>' + cells.map(v => '<td>' + v + '</td>').join('') + '</tr>';
const f3 = v => v.toFixed(3), f1 = v => v.toFixed(1), sg = v => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(1), pc = v => Math.round(100 * v) + '%';
const methods = [
  ['<b>SnowSense v1.0</b>', 'weather only', S.net],
  ['Energy balance, dev-tuned (added on import, seen exam)', 'weather only', RS.eb_devtuned],
  ['Degree-day, dev-tuned (added on import, seen exam)', 'weather only', RS.dd_devtuned],
  ['Energy balance, WaterSim defaults', 'weather only', S.eb_uncal],
  ['Degree-day, WaterSim defaults', 'weather only', S.dd_uncal], ['Climatology (own WY2001-10 SWE)', 'reads station SWE', S.clim],
  ['Degree-day calibrated per station', 'reads station SWE (gauged ceiling)', S.dd_cal],
];
const exam = methods.map(([n, p, m]) => row([n, p, f3(m.nse), f1(m.rmse_mm), f1(m.peak_swe_abs_err_pct), sg(m.peak_swe_bias_pct),
  f1(m.meltout_mae_days), sg(m.meltout_bias_days)])).join('\n');
const regions = Object.entries(S.by_region).sort((a, b) => b[1].n - a[1].n)
  .map(([k, v]) => row([k, v.n, f3(v.net), f3(v.eb_uncal), f3(RSall.by_region[k].eb_devtuned.median_baseline), f3(v.dd_cal)])).join('\n');
const elev = Object.entries(S.by_elev).map(([k, v]) => row([k.replace('-', '–'), v.n, f3(v.net), f3(v.eb_uncal),
  f3(RSall.by_elev[k].eb_devtuned.median_baseline), f3(v.dd_cal)])).join('\n');
const N = S.net;
const cover = [['Daily SWE (snow days)', N.cov68_daily, N.cov90_daily], ['Peak SWE (station-years)', N.peak_cov68, N.peak_cov90],
  ['Melt-out date', N.melt_cov68, N.melt_cov90]].map(([n, a, b]) => row([n, pc(a), pc(b)])).join('\n');
const rungs = R.rungs.map(r => row([r.rung, r.change.split(' (')[0].split(': ')[0], ...['val_p1', 'val_p2', 'val_p3', 'val_p4'].map(k => f3(r.sim[k])),
  f3(r.dev.nse), f3(E[r.rung].nse), f1(E[r.rung].meltout_mae_days),
  r.rung === R.incumbent ? '<b>released</b>' : r.kept ? 'kept' : '<b>rejected</b>'])).join('\n');

const global = await text(join(site, 'src/styles/global.css'));
const tokens = global.slice(global.indexOf(':root {'), global.indexOf('*,'));
const replacements = {
  FONTS: fonts.join('\n'), TOKENS: tokens, INSTRUMENT: await text(join(site, 'src/styles/instrument.css')),
  EXAM: exam, REGIONS: regions, ELEV: elev, COVER: cover, RUNGS: rungs,
  WIN_EB: pc(S.win_frac_vs_eb_uncal), WIN_DD: pc(S.win_frac_vs_dd_cal), WIN_CLIM: pc(S.win_frac_vs_clim),
  PACKS: packs.join('\n'), JS: output.outputFiles[0].text,
};
let html = await text(join(model, 'lab/src/page.html'));
for (const [key, value] of Object.entries(replacements)) html = html.replaceAll('@@' + key + '@@', () => value);
if (html.match(/@@[A-Z0-9_]+@@/)) throw new Error('Unfilled lab placeholder: ' + html.match(/@@[A-Z0-9_]+@@/)[0]);
await writeFile(join(model, 'lab/index.html'), html);
console.log('Built offline SnowSense lab (' + Buffer.byteLength(html) + ' bytes)');
