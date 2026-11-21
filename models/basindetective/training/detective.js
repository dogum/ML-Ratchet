// BasinDetective in the browser: target-period climate features (port of features._climate + the dt_* part of
// featurize) and the numpy MLP ensemble (port of net.Detective). Verified against Python by verify_js.py.
(function (root) {
  'use strict';

  function erf(x) {
    // W. J. Cody-style rational approximation via erfc continued fraction is overkill; use a high-precision
    // series / asymptotic split (abs error < 1e-12 over the range used here).
    var s = x < 0 ? -1 : 1; x = Math.abs(x);
    if (x < 3) {
      var sum = x, term = x, x2 = x * x, n = 0;
      while (Math.abs(term) > 1e-17 * Math.abs(sum) && n < 200) {
        n++; term *= -x2 / n; sum += term / (2 * n + 1);
      }
      return s * 2 / Math.sqrt(Math.PI) * sum;
    }
    // erfc continued fraction (Lentz) for x >= 3
    var tiny = 1e-300, f = x, C = x, Dd = 0, i;
    for (i = 1; i < 300; i++) {
      var an = i / 2;
      Dd = x + an * Dd; if (Math.abs(Dd) < tiny) Dd = tiny; Dd = 1 / Dd;
      C = x + an / C; if (Math.abs(C) < tiny) C = tiny;
      var delta = C * Dd; f *= delta; if (Math.abs(delta - 1) < 1e-16) break;
    }
    var erfc = Math.exp(-x * x) / Math.sqrt(Math.PI) / f;
    return s * (1 - erfc);
  }
  function gelu(x) { return 0.5 * x * (1 + erf(x / Math.SQRT2)); }

  function quantile(sorted, q) {            // numpy 'linear'
    var n = sorted.length, h = (n - 1) * q, lo = Math.floor(h), hi = Math.min(lo + 1, n - 1);
    return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo]);
  }
  function mean(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s / a.length; }
  function std(a) { var m = mean(a), s = 0; for (var i = 0; i < a.length; i++) s += (a[i] - m) * (a[i] - m); return Math.sqrt(s / a.length); }
  function corr(a, b) {
    var ma = mean(a), mb = mean(b), sab = 0, saa = 0, sbb = 0;
    for (var i = 0; i < a.length; i++) { var x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
    return sab / Math.sqrt(saa * sbb);
  }

  // daily arrays P, T, E plus month (1-12) and water year per day
  function climate(P, T, E, month, wy) {
    var n = P.length, pm = Math.max(mean(P), 1e-3), em = Math.max(mean(E), 1e-3);
    var Pm = [], Em = [], Tm = [], m, i;
    for (m = 1; m <= 12; m++) {
      var sp = 0, se = 0, st = 0, c = 0;
      for (i = 0; i < n; i++) if (month[i] === m) { sp += P[i]; se += E[i]; st += T[i]; c++; }
      Pm.push(sp / c / pm); Em.push(se / c / em); Tm.push(st / c);
    }
    var tot = 0, snow = 0, warm = 0, wet = 0;
    for (i = 0; i < n; i++) { tot += P[i]; if (T[i] < 0) snow += P[i]; if (T[i] > 10) warm += P[i]; if (P[i] > 1.0) wet++; }
    var sorted = Array.prototype.slice.call(P).sort(function (a, b) { return a - b; });
    var years = {}, ys = [];
    for (i = 0; i < n; i++) { if (!(wy[i] in years)) { years[wy[i]] = 0; ys.push(wy[i]); } years[wy[i]] += P[i]; }
    var ap = ys.map(function (y) { return years[y]; });
    var pseas = (std(Pm) > 0 && std(Em) > 0) ? corr(Pm, Em) : 0;
    return {
      c: { logP: Math.log(pm), logPET: Math.log(em), arid: em / pm, Tmean: mean(T), fsnow: snow / Math.max(tot, 1e-9),
        fwarm: warm / Math.max(tot, 1e-9), wetfrac: wet / n, p99rel: quantile(sorted, 0.99) / pm, pseas: pseas,
        pcv: ys.length > 1 ? std(ap) / Math.max(mean(ap), 1e-9) : 0 },
      Pm: Pm, Em: Em, Tm: Tm
    };
  }

  var CKEYS = ['logP', 'logPET', 'arid', 'Tmean', 'fsnow', 'fwarm', 'wetfrac', 'p99rel', 'pseas', 'pcv'];

  // Fill the target-period entries of a feature vector x (calibration entries come precomputed from Python).
  // cal: {c: {...}, Pm, Tm} calibration climate (precomputed); tgt: daily {P,T,E,month,wy}; a: P scale; d: T shift (C)
  function targetFeatures(x, names, cal, tgt, a, d) {
    var n = tgt.P.length, P = new Float64Array(n), T = new Float64Array(n), E = new Float64Array(n), i;
    for (i = 0; i < n; i++) { P[i] = tgt.P[i] * a; T[i] = tgt.T[i] + d; E[i] = tgt.E[i] * (1 + 0.025 * d); }
    var tc = climate(P, T, E, tgt.month, tgt.wy), out = x.slice(), idx = {};
    names.forEach(function (nm, j) { idx[nm] = j; });
    CKEYS.forEach(function (k) {
      out[idx['t_' + k]] = tc.c[k];
      if (k !== 'wetfrac') out[idx['dt_' + k]] = tc.c[k] - cal.c[k];
    });
    out[idx['dt_wetfrac']] = tc.c.wetfrac - cal.c.wetfrac;
    for (var m = 0; m < 12; m++) {
      out[idx['dt_Pm' + m]] = tc.Pm[m] * Math.exp(tc.c.logP - cal.c.logP) - cal.Pm[m];
      out[idx['dt_Tm' + m]] = tc.Tm[m] - cal.Tm[m];
    }
    return out;
  }

  // ---- network (ensemble of MLPs; weights as Float32Array row-major)
  function member(W, z) {
    var h = z, l, i, j;
    for (l = 0; l < W.layers.length; l++) {
      var L = W.layers[l], o = new Float64Array(L.nout);
      for (i = 0; i < L.nout; i++) {
        var s = L.b[i], off = i * L.nin;
        for (j = 0; j < L.nin; j++) s += L.w[off + j] * h[j];
        o[i] = gelu(s);
      }
      h = o;
    }
    var O = W.out, out = new Float64Array(O.nout);
    for (i = 0; i < O.nout; i++) { var s2 = O.b[i], off2 = i * O.nin; for (j = 0; j < O.nin; j++) s2 += O.w[off2 + j] * h[j]; out[i] = s2; }
    return out;
  }

  function predict(model, x) {
    var nt = model.targets.length, K = model.members.length, t, k;
    var mus = [], vars_ = [], ps = [];
    for (k = 0; k < K; k++) {
      var W = model.members[k], z = new Float64Array(x.length);
      for (var j = 0; j < x.length; j++) {
        var v = (x[j] - W.x_med[j]) / W.x_iqr[j];
        v = Math.max(-6, Math.min(6, v)); z[j] = isFinite(v) ? v : 0;
      }
      var o = member(W, z), mu = [], vr = [];
      for (t = 0; t < nt; t++) {
        mu.push(o[t] * W.y_sd[t] + W.y_mu[t]);
        var sd = Math.exp(Math.max(-7, Math.min(5, o[nt + t]))) * W.y_sd[t];
        vr.push(sd * sd);
      }
      mus.push(mu); vars_.push(vr); ps.push(1 / (1 + Math.exp(-o[2 * nt])));
    }
    var res = {};
    for (t = 0; t < nt; t++) {
      var m = 0, v2 = 0, sp = 0;
      for (k = 0; k < K; k++) { m += mus[k][t]; v2 += vars_[k][t]; }
      m /= K; v2 /= K;
      for (k = 0; k < K; k++) sp += (mus[k][t] - m) * (mus[k][t] - m);
      sp /= K;
      res[model.targets[t]] = [m, Math.sqrt(v2 + sp) * model.sig_scale[t]];
    }
    var p = mean(ps);
    if (model.cls_temp !== 1) { var lg = Math.log(Math.min(Math.max(p, 1e-9), 1 - 1e-9) / Math.max(1 - p, 1e-9)); p = 1 / (1 + Math.exp(-lg / model.cls_temp)); }
    res.p_regulated = p;
    return res;
  }

  function b64f32(s) {
    var bin = atob(s), n = bin.length, u = new Uint8Array(n);
    for (var i = 0; i < n; i++) u[i] = bin.charCodeAt(i);
    return new Float32Array(u.buffer);
  }
  function b64i16(s) {
    var bin = atob(s), n = bin.length, u = new Uint8Array(n);
    for (var i = 0; i < n; i++) u[i] = bin.charCodeAt(i);
    return new Int16Array(u.buffer);
  }
  function loadModel(J) {
    var f = function (a) { return typeof a === 'string' ? b64f32(a) : a; };
    return {
      targets: J.targets, sig_scale: J.sig_scale, cls_temp: J.cls_temp,
      members: J.members.map(function (M) {
        return { x_med: f(M.x_med), x_iqr: f(M.x_iqr), y_mu: M.y_mu, y_sd: M.y_sd,
          layers: M.layers.map(function (L) { return { nin: L.nin, nout: L.nout, w: f(L.w), b: f(L.b) }; }),
          out: { nin: M.out.nin, nout: M.out.nout, w: f(M.out.w), b: f(M.out.b) } };
      })
    };
  }

  var api = { erf: erf, climate: climate, targetFeatures: targetFeatures, predict: predict, loadModel: loadModel,
    b64f32: b64f32, b64i16: b64i16 };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.BD = api;
})(this);
