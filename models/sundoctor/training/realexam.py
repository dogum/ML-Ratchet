"""Real-data final exam on PVDAQ system 2107 (24 identical inverters, 6 strings each).
The network sees ONE inverter at a time, as a home system would be seen; the 23 siblings supply the answer key."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, pandas as pd, torch
sys.path.insert(0, _P.SRC)
import realdata as R, simbatch as SB, evaluate as E

def load():
    z = np.load(_P.WORK + '/data/sys2107_solar.npz'); days = pd.to_datetime(z['days'], unit='D')
    geo = R.geometry(days, R.SYS2107['lat'], R.SYS2107['tilt'], R.SYS2107['azimuth'])
    return days, z['P'], z['G'], z['T'], z['fr'], geo

def channels(p, g, ta, geo, sl):
    """(len, 96) arrays for one inverter -> (8, len, 96) inputs"""
    return SB.inputs(p[sl], g[sl], ta[sl], geo['cz'][sl], geo['ca'][sl], geo['poa_clear'][sl] / 1000.0)

def run_inverter(net, p, g, ta, geo, ref=None, D=None, stride=None):
    """slide 14-day windows (stride 7), keep the central 7 days of each; ref: (n_days,) reference level or None"""
    D = D or getattr(net, 'days', 14); stride = stride or D // 2
    n = len(p); out = {k: np.full(n, np.nan) for k in ('pf', 'soil', 'frac', 'shade', 'clip', 'outage', 'dfrac', 'step_p', 'pf_sig', 'frac_sig', 'eta0', 'ac_limit')}
    starts = list(range(0, n - D + 1, stride)); Xs = []
    for s in starts:
        X = channels(p, g, ta, geo, slice(s, s + D))
        r = 0.0 if ref is None else ref[s + D // 2]; has = float(ref is not None and np.isfinite(r) and r > 0)
        r = r if has else 0.0
        X = np.concatenate([X, np.full((1,) + X.shape[1:], r, np.float32), np.full((1,) + X.shape[1:], has, np.float32)])
        Xs.append(X)
    P = E.predict(net, np.stack(Xs))
    for w, s in enumerate(starts):
        lo, hi = (0 if s == 0 else D // 4), (D if s + D >= n - stride else D - D // 4)
        for k in out:
            if k in ('eta0', 'ac_limit'): out[k][s + lo:s + hi] = P[k][w]
            else: out[k][s + lo:s + hi] = P[k][w, lo:hi]
    return out

def reference(pf, valid, window=365, q=0.975):
    """causal clean-and-healthy level, matching what training means by the reference: a high percentile of the network's
    own daily pf over the previous year (v1 used the 90th percentile of 180 days, which absorbs long faults)"""
    s = pd.Series(np.where(valid, pf, np.nan))
    return s.rolling(window, min_periods=20).quantile(q).shift(1).values

def exam(net, inverters=range(24), use_ref=True):
    days, P, G, T, fr, geo = load(); res = {'per_inverter': {}}
    rows = []
    for j in inverters:
        p = P[:, :, j].astype(np.float64); valid = (np.isfinite(p) & np.isfinite(G)).mean(1) > 0.5
        o1 = run_inverter(net, p, G, T, geo)
        o = run_inverter(net, p, G, T, geo, reference(o1['pf'], valid & (o1['outage'] < 0.2))) if use_ref else o1
        rows.append((j, o, valid))
    return days, fr, rows

def score(days, fr, rows, n_str=6):
    """compare predicted fraction-online with the sibling answer key, counting dead strings out of 6"""
    K_t, K_p, F_t, F_p, onset_t, onset_p = [], [], [], [], [], []
    for j, o, valid in rows:
        f = fr[:, j]; kt = np.round(n_str * (1 - f)); clean = np.isfinite(f) & (np.abs(n_str * (1 - f) - kt) < 0.25) & valid & (kt >= 0) & (kt < n_str) & (o["outage"] < 0.3)
        kp = np.round(n_str * (1 - np.clip(o['frac'], 0, 1)))
        K_t.append(kt[clean]); K_p.append(kp[clean]); F_t.append(f[clean]); F_p.append(o['frac'][clean])
        # onsets: the dead-string count changes between consecutive clean days
        idx = np.where(clean)[0]; ch = idx[1:][np.diff(kt[idx]) != 0]
        onset_t += [(j, d) for d in ch]; onset_p += [(j, d) for d in np.where(o['step_p'] > 0.5)[0]]
    kt = np.concatenate(K_t); kp = np.concatenate(K_p); ft = np.concatenate(F_t); fp = np.concatenate(F_p)
    conf = np.zeros((4, 4), int)
    for a, b in zip(np.minimum(kt, 3).astype(int), np.minimum(kp, 3).astype(int)): conf[a, b] += 1
    hit = sum(any((j2 == j and abs(d2 - d) <= 1) for j2, d2 in onset_p) for j, d in onset_t)
    return {'n_inverter_days': int(len(kt)), 'count_acc': float(np.mean(kt == kp)), 'count_acc_when_dead': float(np.mean((kt == kp)[kt > 0])) if (kt > 0).any() else None,
            'healthy_called_healthy': float(np.mean(kp[kt == 0] == 0)), 'frac_mae': float(np.mean(np.abs(fp - ft))),
            'confusion_true_rows_0_1_2_3plus': conf.tolist(), 'n_dead_days': int((kt > 0).sum()),
            'onsets_true': len(onset_t), 'onsets_found_pm1day': int(hit), 'step_alarms': len(onset_p)}

if __name__ == '__main__':
    net = E.load(sys.argv[1]); days, fr, rows = exam(net)
    r = score(days, fr, rows); print(json.dumps(r, indent=1))
    json.dump(r, open(sys.argv[1].replace('.pt', '_real.json'), 'w'), indent=1)
