// node verify_js.js data.json -> max |JS - Python| over the example basins and lab settings
const fs = require('fs');
global.atob = (s) => Buffer.from(s, 'base64').toString('binary');
const BD = require('./detective.js');
const D = JSON.parse(fs.readFileSync(process.argv[2] || 'data.json'));
const model = BD.loadModel(D.model);
let worstMu = 0, worstSd = 0, worstP = 0, n = 0;
for (const b of D.basins) {
  const P = BD.b64i16(b.tgt_P), T = BD.b64i16(b.tgt_T), E = BD.b64i16(b.tgt_E);
  const t0 = new Date(b.tgt_start + 'T00:00:00Z');
  const month = new Int32Array(b.n_tgt), wy = new Int32Array(b.n_tgt);
  for (let i = 0; i < b.n_tgt; i++) {
    const d = new Date(t0.getTime() + i * 86400000); const m = d.getUTCMonth() + 1;
    month[i] = m; wy[i] = d.getUTCFullYear() + (m >= 10 ? 1 : 0);
  }
  const tgt = { P: Array.from(P, v => v / 100), T: Array.from(T, v => v / 100), E: Array.from(E, v => v / 100), month, wy };
  const x0 = b.x.map(v => v === null ? NaN : v);
  for (const c of b.check) {
    const x = BD.targetFeatures(x0, D.names, b.cal_clim, tgt, c.a, c.d);
    const r = BD.predict(model, x);
    model.targets.forEach((t, j) => {
      worstMu = Math.max(worstMu, Math.abs(r[t][0] - c.mu[j]) / (Math.abs(c.sd[j]) + 1e-9));
      worstSd = Math.max(worstSd, Math.abs(r[t][1] - c.sd[j]) / (Math.abs(c.sd[j]) + 1e-9));
    });
    worstP = Math.max(worstP, Math.abs(r.p_regulated - c.p));
    n++;
  }
}
console.log(JSON.stringify({ cases: n, max_mu_err_in_sigma: worstMu, max_sd_rel_err: worstSd, max_p_err: worstP }));
