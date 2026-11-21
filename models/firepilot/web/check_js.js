// JS == Python check on the page's embedded inputs: node check_js.js
const fs = require('fs'), zlib = require('zlib');
const FP = require('./fp.js');
const D = JSON.parse(fs.readFileSync(__dirname + '/page_data.json'));
const C = JSON.parse(fs.readFileSync(__dirname + '/page_check.json'));
const un = (b, T) => { const buf = zlib.gunzipSync(Buffer.from(b, 'base64')); return new T(buf.buffer, buf.byteOffset, buf.byteLength / T.BYTES_PER_ELEMENT); };
const COST = [0, 0.6, 0.8, 1.7, 1.3, 1.0, 2.0, 0.6, 1.7, 1.1, 2.0].map(Math.fround);
let worstF = 0, worstM = 0, n = 0;
for (const F of D.fires) {
  const ck = C.find(c => c.id === F.id);
  const g = un(F.blob.g, Uint8Array), aq = un(F.blob.aq, Uint16Array), flq = un(F.blob.flq, Uint8Array);
  const W = Float64Array.from(un(F.blob.W, Float32Array));
  const T = Float64Array.from({ length: F.nT }, (_, k) => k * 0.05);
  const ctx = new FP.Context({ ny: F.ny, nx: F.nx, res: 90, burned0: un(F.blob.b0, Uint8Array),
    arr: Float64Array.from(aq, v => v === 65535 ? NaN : v / 16), fl: Float64Array.from(flq, v => v * 0.05),
    cost: Float64Array.from(g, v => COST[v]), rate: 300, clock_t: T, clock_w: W, head: F.head });
  ck.plans.forEach((p, k) => {
    const f = ctx.features(p); n++;
    f.x.forEach((v, i) => { worstF = Math.max(worstF, Math.abs(v - ck.X[k][i])); });
    worstM = Math.max(worstM, Math.abs(FP.predict(D.net, f.x).mean - ck.m[k]));
  });
  const t0 = Date.now(); const best = FP.plan(ctx, D.net, 7); const dt = Date.now() - t0;
  console.log(F.name, F.ny + 'x' + F.nx, 'browser plan', JSON.stringify(best.p).slice(0, 90), 'pred', best.mean.toFixed(3), dt + ' ms');
}
console.log('plans', n, 'max |feature diff|', worstF, 'max |pred diff|', worstM);
