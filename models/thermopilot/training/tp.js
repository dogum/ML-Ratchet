// ThermoPilot in the browser: a line-by-line port of release/thermopilot_core.py (features, MLP, action menu,
// comfort shield). Checked against Python on stored feature vectors (web/check_js.js).
(function (root) {
  const H = 24, NS = 34, NB = 11, NF = NS + NB * H;
  const KINDS = ['hp', 'furnace', 'elec', 'baseboard', 'none'];
  const DH = [null, 'edge', 0, 1, 2, 3], DC = [null, 'edge', 0, -1, -2, -3];
  const ACTIONS = []; for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) ACTIONS.push([i, j]);
  const SHIELD_RATE = 1.0;

  function interp(x, xs, ys) {
    const n = xs.length;
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
    const t = (x - xs[lo]) / (xs[hi] - xs[lo]);
    return ys[lo] + t * (ys[hi] - ys[lo]);
  }
  function setpoints(loK, hiK, lo, hi, ih, ic) {
    let d = DH[ih];
    let h = d === null ? -100 : (d === 'edge' ? loK : Math.max(lo + d, loK));
    d = DC[ic];
    let c = d === null ? 100 : (d === 'edge' ? hiK : Math.min(hi + d, hiK));
    if (c < h + 0.5) c = h + 0.5;
    return [h, c];
  }
  function prefKW(card, base) {
    let ph = 0;
    if (card.kind === 'hp') ph = interp(-5, card.Tg, card.Ph);
    else if ((card.kind === 'elec' || card.kind === 'baseboard') && !card.gas_heat) ph = interp(-5, card.Tg, card.Qmax);
    const pc = interp(35, card.Tg, card.Pc);
    let mb = 0; for (const v of base) mb += v; mb /= base.length;
    return Math.max(0.5, (Math.max(ph, pc) + card.P_fan + mb) / 1000);
  }
  class Featurizer {
    constructor(card, ctx, lam, base) {
      this.card = card; this.ctx = ctx; this.lam = lam; this.base = base;
      this.P = prefKW(card, base); this.Th = []; this.qh = []; this.peak = new Map();
      this.lo0 = ctx.pref_lo; this.hi0 = ctx.pref_hi; this.ua = Math.max(card.UA, 1);
      const d = new Float64Array(13);
      d[0] = Math.log(this.ua / 150); d[1] = Math.log(Math.max(card.C / this.ua / 3600, 1) / 30);
      d[2] = Math.log(Math.max(card.area, 10) / 150); d[3 + KINDS.indexOf(card.kind)] = 1;
      d[8] = card.fan_cont ? 1 : 0; d[9] = card.res ? 1 : 0; d[10] = card.backup > 0 ? 1 : 0;
      d[11] = card.gas_heat ? 1 : 0; d[12] = Math.log(this.P);
      this.desc = d;
    }
    observe(k, e) {
      const c = this.ctx; if (k <= 0) return; const j = k - 1;
      if (c.dperiod[j] >= 0) { const key = 'd' + c.dperiod[j]; this.peak.set(key, Math.max(this.peak.get(key) || 0, e)); }
      if (c.frate[j] > 0) { this.peak.set('f', Math.max(this.peak.get('f') || 0, e)); }
    }
    step(k, Tz, q, e, fcT, fcG) {
      const c = this.ctx, card = this.card; this.observe(k, e);
      this.Th.push(Tz); this.qh.push(q);
      const n = c.price.length, idx = [];
      for (let j = 0; j < H; j++) idx.push(Math.min(k + j, n - 1));
      const price = idx.map(i => c.price[i]);
      let pm = 0; for (const p of price) pm += Math.abs(p); pm = Math.max(pm / H, 0.02);
      const lo = idx.map(i => c.lo[i]), hi = idx.map(i => c.hi[i]);
      const x = new Float64Array(NF);
      x[0] = (Tz - lo[0]) / 3; x[1] = (hi[0] - Tz) / 3; x[2] = (Tz - this.lo0) / 3; x[3] = (this.hi0 - Tz) / 3;
      for (let i = 1; i < 4; i++) x[3 + i] = this.Th.length > i ? Tz - this.Th[this.Th.length - 1 - i] : 0;
      for (let i = 1; i < 4; i++) x[6 + i] = this.qh.length >= i ? this.qh[this.qh.length - i] / this.ua / 10 : 0;
      x[10] = e / this.P;
      x[11] = c.dperiod[k] >= 0 ? (this.peak.get('d' + c.dperiod[k]) || 0) / this.P : 0;
      x[12] = c.frate[k] > 0 ? (this.peak.get('f') || 0) / this.P : 0;
      const h = c.hour[k];
      x[13] = Math.sin(2 * Math.PI * h / 24); x[14] = Math.cos(2 * Math.PI * h / 24); x[15] = c.wd[k] ? 1 : 0;
      x[16] = Math.log(Math.max(price[0], 0.005) / 0.15);
      x[17] = Math.log(this.lam / (pm * this.P));
      x[18] = c.drate[k] / Math.max(c.drem[k], 1) / pm / 10;
      x[19] = c.frate[k] / Math.max(c.frem[k], 1) / pm / 10;
      x[20] = c.state[k] === 0 ? 1 : 0;
      for (let i = 0; i < 13; i++) x[21 + i] = this.desc[i];
      let b = NS;
      for (let j = 0; j < H; j++) {
        const To = fcT[j], G = fcG[j], i = idx[j];
        const pc = interp(To, card.Tg, card.Pc), qs = interp(To, card.Tg, card.Qs);
        x[b + j] = (To - Tz) / 10;
        x[b + H + j] = G / 800;
        x[b + 2 * H + j] = price[j] / pm - 1;
        x[b + 3 * H + j] = (lo[j] - this.lo0) / 3;
        x[b + 4 * H + j] = (hi[j] - this.hi0) / 3;
        x[b + 5 * H + j] = interp(To, card.Tg, card.eff) / 4;
        x[b + 6 * H + j] = pc > 0 ? qs / Math.max(pc, 1e-9) / 4 : 0;
        x[b + 7 * H + j] = this.base[i] / 1000 / this.P;
        x[b + 8 * H + j] = (c.drate[i] > 0 ? 1 : 0) + 0.5 * (c.frate[i] > 0 ? 1 : 0);
        x[b + 9 * H + j] = interp(To, card.Tg, card.Qmax) / this.ua / 20;
        x[b + 10 * H + j] = qs / this.ua / 20;
      }
      return x;
    }
  }
  function mlp(W, x) {
    // W: {mu, sd, layers: [{W: Float32Array (in*out, row-major in x out), b, nin, nout}]}
    let h = new Float64Array(x.length);
    for (let i = 0; i < x.length; i++) h[i] = (Math.fround(x[i]) - W.mu[i]) / W.sd[i];
    for (let l = 0; l < W.layers.length; l++) {
      const L = W.layers[l], out = new Float64Array(L.nout);
      for (let o = 0; o < L.nout; o++) out[o] = L.b[o];
      for (let i = 0; i < L.nin; i++) {
        const hi = h[i]; if (hi === 0) continue;
        const row = i * L.nout;
        for (let o = 0; o < L.nout; o++) out[o] += hi * L.W[row + o];
      }
      if (l < W.layers.length - 1) for (let o = 0; o < L.nout; o++) if (out[o] < 0) out[o] = 0;
      h = out;
    }
    return h;
  }
  class Pilot {
    constructor(W, card, ctx, lam, base) { this.W = W; this.ctx = ctx; this.feat = new Featurizer(card, ctx, lam, base); }
    step(k, Tz, q, e, fcT, fcG) {
      const x = this.feat.step(k, Tz, q, e, fcT, fcG);
      const lg = mlp(this.W, x);
      let a = 0; for (let i = 1; i < lg.length; i++) if (lg[i] > lg[a]) a = i;
      const c = this.ctx;
      let [h, cs] = setpoints(c.lo[k], c.hi[k], c.pref_lo, c.pref_hi, ACTIONS[a][0], ACTIONS[a][1]);
      const sh = this.W.shield || 0;
      if (sh) {
        let lo = c.lo[k], hi = c.hi[k]; const n = c.lo.length;
        if (sh === 2) { const k1 = Math.min(k + 1, n - 1); lo = Math.max(lo, c.lo[k1]); hi = Math.min(hi, c.hi[k1]); }
        else if (sh >= 4) for (let j = 1; j < 7; j++) {
          const kj = Math.min(k + j, n - 1);
          lo = Math.max(lo, c.lo[kj] - SHIELD_RATE * (j - 1)); hi = Math.min(hi, c.hi[kj] + 0.5 * SHIELD_RATE * (j - 1));
        }
        else if (sh >= 3) for (let j = 1; j < 5; j++) {
          const kj = Math.min(k + j, n - 1);
          lo = Math.max(lo, c.lo[kj] - SHIELD_RATE * (j - 1)); hi = Math.min(hi, c.hi[kj] + SHIELD_RATE * (j - 1));
        }
        h = Math.max(h, lo); cs = Math.min(cs, hi); if (cs < h + 0.5) cs = h + 0.5;
      }
      this.lastAction = a;
      return [h, cs, x, lg];
    }
  }
  function loadWeights(obj) {
    // obj: {mu: [..], sd: [..], layers: [{W: base64 float32, b: [..], nin, nout}], shield}
    const dec = s => { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return new Float32Array(u.buffer); };
    return { mu: Float64Array.from(obj.mu), sd: Float64Array.from(obj.sd), shield: obj.shield || 0,
      layers: obj.layers.map(L => ({ W: dec(L.W), b: Float64Array.from(L.b), nin: L.nin, nout: L.nout })) };
  }
  root.TP = { Featurizer, Pilot, mlp, setpoints, interp, loadWeights, ACTIONS, NF };
})(typeof window !== 'undefined' ? window : globalThis);
