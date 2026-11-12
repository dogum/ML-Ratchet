"""Real-data exams for SunDoctor v2: PVDAQ 2107 (fixed tilt, California) with and without its irradiance sensor, and
PVDAQ 9068 (single-axis trackers, Colorado: snow, stuck trackers, outages), always one inverter (module) at a time."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, _P.SRC)
import simbatch2 as SB2, evaluate2 as E2, realexam as RX, real9068 as R9

def run_system(net, p, g, ta, geo, ref=None, home=False, D=None):
    """slide windows over one system's record; g ignored in home mode. Returns per-day outputs."""
    D = D or getattr(net, 'days', 14); stride = D // 2; n = len(p)
    keys = ['pf', 'soil', 'frac', 'shade', 'clip', 'outage', 'dfrac', 'snow', 'stuck', 'step_p', 'pf_sig', 'frac_sig', 'snow_sig', 'stuck_sig', 'outage_sig']
    out = {k: np.full(n, np.nan) for k in keys + ['eta0', 'ac_limit']}
    starts = list(range(0, n - D + 1, stride)); Xs = []
    for s in starts:
        sl = slice(s, s + D)
        X = SB2.inputs(p[sl], None if home else g[sl], ta[sl], geo['cz'][sl], geo['ca'][sl], geo['poa_clear'][sl] / 1000.0, not home)
        r = 0.0 if ref is None else ref[s + D // 2]; has = float(ref is not None and np.isfinite(r) and r > 0)
        Xs.append(np.concatenate([X, np.full((1,) + X.shape[1:], r if has else 0.0, np.float32), np.full((1,) + X.shape[1:], has, np.float32)]))
    P = E2.predict(net, np.stack(Xs))
    for w, s in enumerate(starts):
        lo, hi = (0 if s == 0 else D // 4), (D if s + D >= n - stride else D - D // 4)
        for k in out:
            out[k][s + lo:s + hi] = P[k][w] if k in ('eta0', 'ac_limit') else P[k][w, lo:hi]
    return out

def diagnose(net, p, g, ta, geo, home=False):
    valid = (np.isfinite(p) & (True if home else np.isfinite(g))).mean(1) > 0.5
    o1 = run_system(net, p, g, ta, geo, None, home); ref = RX.reference(o1['pf'], valid & (o1['outage'] < 0.2))
    o = run_system(net, p, g, ta, geo, ref, home); o['ref'] = ref; o['valid'] = valid; return o

def exam_2107(net, home=False):
    days, P, G, T, fr, geo = RX.load(); rows = []
    for j in range(24):
        p = P[:, :, j].astype(np.float64); o = diagnose(net, p, G, T, geo, home); rows.append((j, o, o['valid']))
    return RX.score(days, fr, rows), rows

def auc(s, y):
    m = np.isfinite(s) & np.isfinite(y); return E2.auc(s[m], y[m].astype(float)) if m.sum() and 0 < y[m].sum() < m.sum() else None

def exam_9068(net, home=False):
    days, d, geo = R9.load(); K = np.load(_P.WORK + '/data/sys9068_keys.npz'); P, G, T = d['P'].astype(np.float64), d['G'], d['T']
    res = {'stuck': [], 'outage': [], 'snow': []}; per = []
    for j in range(8):
        Gj = G if home else np.where(K['sensor_ok'][:, None], G, np.nan)       # with-sensor mode only where the sensor works
        o = diagnose(net, P[:, :, j], Gj, T, geo, home); per.append(o); inv = 1 if j < 4 else 2
        st = K['stuck%d' % inv]; v = o['valid'] & np.isfinite(st) & (np.nan_to_num(K['outage'][:, j]) < 0.2) & ~(K['snow'] > 0).any(1)
        res['stuck'].append(dict(pred=o['stuck'][v], true=st[v]))
        vo = (o['valid'] | home) & np.isfinite(K['outage'][:, j]); res['outage'].append(dict(pred=o['outage'][vo], true=np.nan_to_num(K['outage'][:, j])[vo]))
        vs = K['snow_known'][:, j] & (np.nan_to_num(K['outage'][:, j]) < 0.2) & o['valid']; res['snow'].append(dict(pred=o['snow'][vs], true=K['snow'][vs, j], month=days.month.values[vs], out=o['outage'][vs]))
    cat = lambda k, f: np.concatenate([r[f] for r in res[k]])
    sp, stt = cat('stuck', 'pred'), cat('stuck', 'true'); op, ot = cat('outage', 'pred'), cat('outage', 'true'); snp, snt, snm, op_s = cat('snow', 'pred'), cat('snow', 'true'), cat('snow', 'month'), cat('snow', 'out')
    S = {'stuck_corr': float(np.corrcoef(np.nan_to_num(sp), stt)[0, 1]), 'stuck_auc_2pct': auc(sp, (stt > 0.02).astype(float)), 'stuck_mae': float(np.nanmean(np.abs(sp - stt))),
         'stuck_true_mean': float(np.mean(stt)), 'stuck_pred_mean': float(np.nanmean(sp)), 'stuck_days_big': int((stt > 0.1).sum()), 'stuck_found_big': float(np.nanmean(sp[stt > 0.1] > 0.05)) if (stt > 0.1).any() else None,
         'outage_auc': auc(op, (ot > 0.5).astype(float)), 'outage_found': float(np.nanmean(op[ot > 0.5] > 0.5)), 'outage_false': float(np.nanmean(op[ot < 0.02] > 0.5)), 'outage_days': int((ot > 0.5).sum()),
         'snow_days': int((snt > 0.2).sum()), 'snow_found': float(np.nanmean(snp[snt > 0.2] > 0.2)) if (snt > 0.2).any() else None, 'snow_auc': auc(snp, (snt > 0.2).astype(float)),
         'snow_mae_on_snow_days': float(np.nanmean(np.abs(snp[snt > 0.2] - snt[snt > 0.2]))) if (snt > 0.2).any() else None,
         'snow_called_outage': float(np.nanmean(op_s[snt > 0.2] > 0.5)) if (snt > 0.2).any() else None,
         'snow_false_summer': float(np.nanmean(snp[np.isin(snm, [5, 6, 7, 8, 9])] > 0.2)), 'snow_false_winter_clear': float(np.nanmean(snp[(snt == 0) & np.isin(snm, [11, 12, 1, 2, 3])] > 0.2))}
    return S, per

if __name__ == '__main__':
    net = E2.load(sys.argv[1]); out = {}
    for home in (False, True):
        r, _ = exam_2107(net, home); out['2107_' + ('home' if home else 'poa')] = r
        print('2107', 'home' if home else 'poa', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != 'confusion_true_rows_0_1_2_3plus'}, flush=True)
        s, _ = exam_9068(net, home); out['9068_' + ('home' if home else 'poa')] = s
        print('9068', 'home' if home else 'poa', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
    json.dump(out, open(sys.argv[1].replace('.pt', '_real2.json'), 'w'), indent=1)
