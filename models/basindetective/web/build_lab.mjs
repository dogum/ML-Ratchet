// Bundle the site's BasinDetective instrument into one offline HTML file (lab/index.html), with the exam tables
// generated from training/exam.json, training/ratchet.json and training/twin_final.json, and the B baselines.
// Intervals as in training/import_check.py: Fisher z for rho, Zou (2007) for a difference of two rhos treated as
// independent (conservative), Hanley-McNeil for AUC, exact binomial for coverage.
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const model = fileURLToPath(new URL('../', import.meta.url));
const site = join(model, '../../site');
const require = createRequire(join(site, 'package.json'));
const { build } = require('esbuild');
const text = path => readFile(path, 'utf8');
const packDir = join(site, 'public/instruments/basindetective');
const output = await build({ entryPoints: [join(model, 'lab/src/entry.ts')], bundle: true,
  format: 'iife', minify: true, write: false, target: 'es2022' });

const packs = [];
for (const name of ['weights', 'examples']) {
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

// ---- statistics
const fisher = (r, n) => { const z = Math.atanh(r), se = 1 / Math.sqrt(n - 3); return [Math.tanh(z - 1.96 * se), Math.tanh(z + 1.96 * se)]; };
const zou = (r1, r2, n) => {
  const [l1, u1] = fisher(r1, n), [l2, u2] = fisher(r2, n), d = r1 - r2;
  return [d, d - Math.hypot(r1 - l1, u2 - r2), d + Math.hypot(u1 - r1, r2 - l2)];
};
const hanley = (a, np, nn) => {
  const q1 = a / (2 - a), q2 = 2 * a * a / (1 + a);
  const se = Math.sqrt((a * (1 - a) + (np - 1) * (q1 - a * a) + (nn - 1) * (q2 - a * a)) / (np * nn));
  return [a - 1.96 * se, a + 1.96 * se];
};
const lgam = x => { // Lanczos
  const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgam(1 - x);
  x -= 1; let a = c[0]; const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
};
const binomCdf = (k, n, p) => { let s = 0; for (let i = 0; i <= k; i++) s += Math.exp(lgam(n + 1) - lgam(i + 1) - lgam(n - i + 1) + i * Math.log(p) + (n - i) * Math.log1p(-p)); return s; };
const bisect = (f, lo, hi) => { for (let i = 0; i < 80; i++) { const m = (lo + hi) / 2; if (f(m)) hi = m; else lo = m; } return (lo + hi) / 2; };
const clopper = (k, n) => [k === 0 ? 0 : bisect(p => 1 - binomCdf(k - 1, n, p) >= 0.025, 0, 1), k === n ? 1 : bisect(p => binomCdf(k, n, p) <= 0.025, 0, 1)];
const verdict = (lo, hi) => (lo > 0 ? 'win' : hi < 0 ? 'loss' : 'tie');
const f2 = v => (v == null ? '–' : v.toFixed(2)), f3 = v => v.toFixed(3);
const sg = v => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(3);
const ci = ([a, b], f = f2) => `<small>[${f(a)}, ${f(b)}]</small>`;
const tr = cells => '<tr>' + cells.map(v => '<td>' + v + '</td>').join('') + '</tr>';

const ex = JSON.parse(await text(join(model, 'training/exam.json')));
const NICE = { d_logq: 'Mean flow (log ratio)', d_rr: 'Runoff ratio', d_bfi: 'Baseflow index (LH)', d_logq5: 'Low flow Q5 (log ratio)',
  d_logq95: 'High flow Q95 (log ratio)', d_hfd: 'Half-flow date (days)' };
const G = 'GR4J (gauged, model-implied change)', G2 = 'GR4J (gauged, sim target period minus observed cal)', BU = 'Budyko-Fu (w fitted on cal)';
const changes = Object.entries(ex.delta).map(([t, d]) => {
  const m = d.BasinDetective, n = m.n;
  const base = [G, G2, BU].filter(k => d[k]).map(k => [k, d[k].rho]).sort((a, b) => b[1] - a[1]);
  const [diff, lo, hi] = zou(m.rho, base[0][1], n);
  const dg = t === 'd_hfd' ? v => v.toFixed(1) : f3;
  const cov = k => (100 * m[k]).toFixed(0) + '%';
  return tr([NICE[t], `<b>${f2(m.rho)}</b> ${ci(fisher(m.rho, n))}`, f2(d[G].rho), `${sg(diff)} <small>[${sg(lo)}, ${sg(hi)}] vs ${base[0][0].startsWith('Budyko') ? 'Budyko' : base[0][0] === G2 ? 'GR4J/own' : 'GR4J'}</small>`,
    f2(d[G2].rho), d[BU] ? f2(d[BU].rho) : '–', `${dg(m.mae)} / ${dg(d[G].mae)} / ${dg(d['no change (persistence)'].mae)}`,
    `${cov('cov68')} ${ci(clopper(Math.round(m.cov68 * n), n), v => (100 * v).toFixed(0))} / ${cov('cov95')}`, verdict(lo, hi)]);
});

const bsum = JSON.parse((await text(join(model, '../../shared/water/data/proc/baselines/detective_summary.json')).catch(() => '{}')).replaceAll('NaN', 'null'));
const levels = Object.entries(ex.level).map(([lv, L]) => {
  const m = L.BasinDetective, own = L['own calibration-period value'], gr = L['GR4J sim (gauged)'];
  const best = own.rho > gr.rho ? own : gr;
  const [d, lo, hi] = zou(m.rho, best.rho, m.n);
  const knn = bsum[`${lv}/knn_attr`]?.spearman;
  return tr([lv, `<b>${f3(m.rho)}</b>`, f3(own.rho), f3(gr.rho), knn != null ? f3(knn) : '–', `${sg(d)} <small>[${sg(lo)}, ${sg(hi)}]</small>`, verdict(lo, hi)]);
});

const R = ex.regulation;
const reg = Object.entries(R).filter(([, r]) => 'n_pos' in r).map(([name, r]) => {
  const np = r.n_pos, nn = ex.n_basins - np;
  const a = k => `${f2(r[k])} ${ci(hanley(r[k], np, nn))}`;
  return tr([name, np, '<b>' + a('BasinDetective_auc') + '</b>', a('GR4J misfit (1-NSE_cal)_auc'), a('seasonal damping index_auc'), f2(r.BasinDetective_precision_at_n), 'tie']);
});
const ld = R['log10 dor among regulated basins'], cf = R['confusion_p>0.5'];
const regNote = `AUC intervals: Hanley-McNeil (approximate, unpaired), so every row is a tie; the three HydroATLAS rows include 0.5, chance. `
  + `Simulated AUC on the frozen set: 0.91. Among the ${ld.n} exam basins with any upstream reservoir, log10 storage / annual flow: ρ ${f2(ld.rho)} ${ci(fisher(ld.rho, ld.n))}, `
  + `median |error| ${f2(ld.mae)}, 68 % coverage ${(100 * ld.cov68).toFixed(0)}% (σ not calibrated: no dev truth). At p > 0.5: ${cf.tp} hits, ${cf.fp} false alarms, ${cf.fn} misses, ${cf.tn} correct rejections.`;

const tw = JSON.parse(await text(join(model, 'training/twin_final.json'))).param_recovery;
const hidden = Object.entries(tw).map(([k, v]) => tr([k, v.rho == null ? '–' : `${f2(v.rho)} ${ci(fisher(v.rho, 260))}`, v.r2 < -1 ? '–' : f2(v.r2), f3(v.median_abs_err), sg(v.bias), (100 * v.cov68).toFixed(0) + '%']));

const strata = Object.entries(ex.strata).map(([name, s]) => tr([name, s.n, ...['d_logq', 'd_rr', 'd_bfi', 'd_logq5', 'd_logq95', 'd_hfd'].map(t => `${f2(s[t].net)} / ${f2(s[t].gr4j)}`)]));

const rj = JSON.parse(await text(join(model, 'training/ratchet.json')));
const v4 = rj.rescores.find(r => r.val === 'v4').scores;
const rungs = rj.rungs.map(r => tr([r.name, `<span style="white-space:normal;display:inline-block;max-width:44ch;text-align:left">${r.description.replaceAll('<', '&lt;')}</span>`, r.physics, f3(r.sim_skill),
  r.physics === 'v4' ? f3(r.sim_skill) : f3(v4[r.name].sim_skill), f3(r.dev_composite), r.name === 'R5' ? '<b>released</b>' : r.kept ? 'kept' : 'rejected']));

const global = await text(join(site, 'src/styles/global.css'));
const tokens = global.slice(global.indexOf(':root {'), global.indexOf('*,'));
const replacements = {
  FONTS: fonts.join('\n'), TOKENS: tokens, INSTRUMENT: await text(join(site, 'src/styles/instrument.css')),
  CHANGES: changes.join('\n'), LEVELS: levels.join('\n'), REG: reg.join('\n'), REGNOTE: regNote, HIDDEN: hidden.join('\n'),
  STRATA: strata.join('\n'), RUNGS: rungs.join('\n'), PACKS: packs.join('\n'), JS: output.outputFiles[0].text,
};
let html = await text(join(model, 'lab/src/page.html'));
for (const [key, value] of Object.entries(replacements)) html = html.replace('@@' + key + '@@', () => value);
if (html.match(/@@[A-Z0-9]+@@/)) throw new Error('Unfilled lab placeholder');
await writeFile(join(model, 'lab/index.html'), html);
console.log('Built offline BasinDetective lab (' + Buffer.byteLength(html) + ' bytes)');
