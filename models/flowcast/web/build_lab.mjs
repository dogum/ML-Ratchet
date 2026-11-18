// Bundle the site's FlowCast instrument, its packs, fonts and the shipped exam and ratchet results into one offline
// HTML file: models/flowcast/lab/index.html. Run from anywhere: node models/flowcast/web/build_lab.mjs
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
const packDir = join(site, 'public/instruments/flowcast');
const output = await build({ entryPoints: [join(model, 'lab/src/entry.ts')], bundle: true,
  format: 'iife', minify: true, write: false, target: 'es2022' });

const packs = [];
for (const name of ['weights', 'nowcast', 'basins']) {
  const manifest = JSON.parse(await text(join(packDir, name + '.json')));
  const bytes = (await readFile(join(packDir, name + '.bin'))).toString('base64');
  const safe = JSON.stringify({ manifest, bytes }).replaceAll('<', '\\u003c');
  packs.push('<script type="application/json" id="pack-' + name + '">' + safe + '</script>');
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

// ---- tables from the shipped result JSONs (training/results, training/ratchet.json)
const res = join(model, 'training/results');
const exam = await json(join(res, 'exam_r1b_exam_cal.json'));
const nowc = await json(join(res, 'nowcast_summary.json'));
const ratchet = await json(join(model, 'training/ratchet.json'));
const row = (cells, cls = '') => '<tr' + (cls ? ' class="' + cls + '"' : '') + '>' + cells.map(v => '<td>' + v + '</td>').join('') + '</tr>';
const f3 = v => (v == null ? '—' : (v < 0 ? '−' : '') + Math.abs(v).toFixed(3));
const f0 = v => (v < 0 ? '−' : v > 0 ? '+' : '') + Math.abs(v).toFixed(0);
const pc = v => Math.round(100 * v) + '%';

const methods = [['flowcast', 'FlowCast v1.0', 'ungauged'], ['nn_dev', 'Regionalisation, dev donors (U-dev)', 'ungauged'],
  ['nn_k12', 'Regionalisation, 12-fold (U-k12)', 'ungauged'], ['gr4j', 'GR4J + snow, calibrated per basin', 'gauged'],
  ['nwm', 'NOAA National Water Model v3.0', 'operational'], ['persist1', 'Persistence, 1 day', 'gauged forecast'],
  ['clim', 'Climatology', 'gauged']];
const fl = exam.flow;
const examRows = methods.map(([k, name, proto]) => {
  const v = fl['val/' + k], r = fl['recent/' + k];
  const win = k === 'flowcast' ? '—' : pc(v.flowcast_wins_frac);
  return row([k === 'flowcast' ? '<b>' + name + '</b>' : name, proto, f3(v.nse), f3(v.kge), f0(v.fhv) + '%', pc(v.frac_nse_lt0),
    f3(r.nse), f3(r.kge), win], k === 'flowcast' ? 'me' : '');
});
const strata = Object.entries(exam.strata).filter(([k]) => k.startsWith('val/')).map(([k, v]) => {
  const r = exam.strata['recent/' + k.slice(4)];
  return row([k.slice(4), v.n, '<b>' + v.flowcast.toFixed(2) + '</b>', v.nn_dev.toFixed(2), v.gr4j.toFixed(2), v.nwm.toFixed(2),
    r.flowcast.toFixed(2) + ' / ' + r.nn_dev.toFixed(2) + ' / ' + r.nwm.toFixed(2)]);
});
const nowRows = [1, 3, 7].map(l => {
  const v = nowc.exam['val/' + l], r = nowc.exam['recent/' + l], d = nowc.dev[String(l)];
  return row([l + (l === 1 ? ' day' : ' days'), f3(d.nse) + ' / ' + f3(d.nse_persist), '<b>' + f3(v.nse) + '</b>', f3(v.nse_persist), pc(v.wins_vs_persist),
    '<b>' + f3(r.nse) + '</b>', f3(r.nse_persist), pc(r.wins_vs_persist)]);
});
const pk = exam.peaks, pb = pk.baselines_B;
const peakRows = [['FlowCast mean', pk['val/mean'], pk['recent/mean']], ['FlowCast q95', pk['val/q95'], pk['recent/q95']],
  ['Own-record median (gauged)', pb['val/cal_median'], pb['recent/cal_median']], ['GR4J calibrated × 1.38', pb['val/gr4j_ratio'], pb['recent/gr4j_ratio']],
  ['NWM daily × 1.38', pb['val/nwm_ratio'], pb['recent/nwm_ratio']], ['NWM hourly', null, pb['recent/nwm_hourly']]]
  .map(([n, v, r]) => row([n, v ? Math.round(v.median_abs_pct_err) + '%' : '—', v ? f0(v.median_bias_pct) + '%' : '—',
    Math.round(r.median_abs_pct_err) + '%', f0(r.median_bias_pct) + '%']));

const sets = ['p0', 'p1', 'p2', 'p3', 'p4'], rungIds = ['r0', 'r1', 'r0b', 'r1b', 'r2', 'r3', 'r4', 'r5'];
const sim = {};
for (const r of rungIds) for (const p of sets)   // the sim JSONs hold NaN for flat simulated records
  sim[r + p] = JSON.parse((await text(join(res, `sim_${r}_${p}.json`))).replace(/\bNaN\b/g, 'null')).nse_median;
const byId = Object.fromEntries(ratchet.rungs.map(r => [r.rung, r]));
const r0dev = byId.r1.dev_incumbent;
const label = { r0: 'WaterSim as shipped', r1: 'Water-yield prior p1, old loss', r0b: 'Loss scaled by the Fu runoff ratio of the forcing',
  r1b: 'Water-yield prior p1 + new loss', r2: 'Flashier runoff (full dev prior shift)', r3: 'Realistic forcing error',
  r4: 'Forcing error + flashier runoff', r5: 'r1b, 5,000 more steps' };
const ladder = rungIds.map(id => {
  const r = byId[id], dev = id === 'r0' ? r0dev : r.dev;
  const verdict = id === 'r0' ? 'base' : id === 'r1b' ? '<b>kept, released</b>' : r.kept ? 'kept' : 'rejected';
  const own = id === 'r0' ? 'p0' : r.frozen_set;
  return row([id, label[id], own, f3(sim[id + own]) + (id === 'r0' ? '' : ' vs ' + f3(r.sim_nse_incumbent)), f3(dev.nse_median),
    f3(dev.kge_median), f0(dev.fhv_median) + '%', verdict], id === 'r1b' ? 'me' : '');
});
const matrix = rungIds.map(id => row([id, ...sets.map(p => {
  const best = Math.max(...rungIds.map(r => sim[r + p]));
  return sim[id + p] === best ? '<b>' + f3(sim[id + p]) + '</b>' : f3(sim[id + p]);
})], id === 'r1b' ? 'me' : ''));

const global = await text(join(site, 'src/styles/global.css'));
const tokens = global.slice(global.indexOf(':root {'), global.indexOf('*,'));
const replacements = {
  FONTS: fonts.join('\n'), TOKENS: tokens, INSTRUMENT: await text(join(site, 'src/styles/instrument.css')),
  EXAM: examRows.join('\n'), STRATA: strata.join('\n'), NOWCAST: nowRows.join('\n'), PEAKS: peakRows.join('\n'),
  LADDER: ladder.join('\n'), MATRIX: matrix.join('\n'), PACKS: packs.join('\n'), JS: output.outputFiles[0].text,
};
let html = await text(join(model, 'lab/src/page.html'));
for (const [key, value] of Object.entries(replacements)) html = html.replace('@@' + key + '@@', () => value);
if (html.match(/@@[A-Z0-9]+@@/)) throw new Error('Unfilled lab placeholder');
await writeFile(join(model, 'lab/index.html'), html);
console.log('Built offline FlowCast lab (' + Buffer.byteLength(html) + ' bytes)');
