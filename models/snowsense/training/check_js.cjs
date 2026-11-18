// JS network vs the numpy release on the explorer windows (py_swe precomputed by webdata.py with the numpy model)
const fs = require('fs');
const SS = require('./snowsense.js');
const W = JSON.parse(fs.readFileSync(__dirname + '/weights.json'));
const rnd = x => Array.isArray(x) ? x.map(rnd) : +x.toPrecision(6);   // as in build.py
for (const k in W.weights) W.weights[k] = rnd(W.weights[k]);
const D = JSON.parse(fs.readFileSync(__dirname + '/data.json'));
let worst = 0, n = 0;
for (const st of D.stations.slice(0, 4)) for (const w of st.wins.slice(-2)) {
  const mP = new Array(427).fill(0), mT = new Array(427).fill(0);
  w.mP.forEach(i => mP[i] = 1); w.mT.forEach(i => mT[i] = 1);
  const r = SS.run(W, { ...w, mP, mT, lat: st.lat, elev: st.elev });
  for (let i = 0; i < w.nwy; i++) { worst = Math.max(worst, Math.abs(r.swe[i] - w.py_swe[i])); n++; }
}
const out = { windows: 8, days: n, max_abs_swe_mm_vs_numpy: worst, note: 'inputs rounded to 0.1 as on the page; weights rounded to 6 significant digits as on the page' };
fs.writeFileSync(__dirname + '/../results/js_check.json', JSON.stringify(out, null, 1));
console.log(out);
