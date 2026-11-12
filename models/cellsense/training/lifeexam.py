"""CellSense final exam, lifetime: forecast each real cell's cycle life (first cycle below 0.88 Ah) from one reading.
Severson et al. used the first 100 cycles; their split is reproduced here (cells ordered batch 1, 2, 3 after their own
exclusions: training = odd indices of batches 1-2, primary test = even indices plus the last of batch 2, secondary test =
batch 3). CellSense trained on none of them; the calibration cells are left out wherever they fall."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import realdata as RD, realexam as RX, calibrate as CA
STEP = 4
SEVERSON = {'variance': {'primary_rmse': 138, 'secondary_rmse': 196, 'primary_mpe': 14.7, 'secondary_mpe': 11.4},
            'discharge': {'primary_rmse': 91, 'secondary_rmse': 173, 'primary_mpe': 13.0, 'secondary_mpe': 8.6},
            'full': {'primary_rmse': 118, 'secondary_rmse': 214, 'primary_mpe': 14.1, 'secondary_mpe': 10.7}}   # Table 2 of the paper (RMSE in cycles)

def split(cells):
    keys = list(cells.keys()); n12 = sum(cells[k]['batch'] in ('b1', 'b2') for k in keys); S = {}
    for i, k in enumerate(keys):
        S[k] = 'secondary' if i >= n12 else ('primary' if (i % 2 == 0 or i == n12 - 1) else 'train')
    return S

def readings(cells, cycles=(100,), every=None, only=None):
    """only=None: exam cells (calibration cells left out); only=list: just those cells (the dev set)"""
    rows = []; SP = split(cells)
    for k, c in cells.items():
        if (only is None and k in CA.CAL) or (only is not None and k not in only): continue
        S = c['summary']; idx = c['cyc_idx']; life = c['cycle_life_pub']
        ref_rows = [i for i, j in enumerate(idx) if 4 <= j <= 9]
        q0 = np.nanmean(c['Qdlin'][ref_rows], 0)[::STEP]; t0 = np.nanmean(c['Tdlin'][ref_rows], 0)[::STEP]
        ct = S['chargetime'][1:51]; ct = float(np.nanmedian(ct[(ct > 5) & (ct < 30)]))
        want = list(cycles) + (list(range(every, int(life * 1.05), every)) if every else [])
        for n in want:
            r = int(np.argmin(np.abs(idx + 1 - n)))
            if abs(idx[r] + 1 - n) > 3: continue
            q = c['Qdlin'][r][::STEP]; t = c['Tdlin'][r][::STEP]
            if not (np.isfinite(q).all() and np.isfinite(t).all() and 0.5 < np.nanmax(q) < 1.25 and np.nanmin(q) > -0.01 and 15 < t.min() and t.max() < 60): continue
            rows.append(dict(cell=k, split=SP[k], cycle=int(idx[r] + 1), life=float(life), charge_time=ct, Q1=q, T1=t, Q0=q0, T0=t0))
    return rows

def score(rows, P, tag=''):
    life = np.array([r['life'] for r in rows]); pred = 10 ** P['log_life']; sig = P['log_life_sig']; sp = np.array([r['split'] for r in rows]); out = {}
    for name, m in (('primary', sp == 'primary'), ('secondary', sp == 'secondary'), ('all', np.ones(len(rows), bool))):
        if not m.any(): continue
        e = pred[m] - life[m]
        out[name] = {'n': int(m.sum()), 'rmse': float(np.sqrt(np.mean(e ** 2))), 'mpe': float(100 * np.mean(np.abs(e) / life[m])), 'bias': float(np.mean(e)),
                     'corr_log': float(np.corrcoef(np.log10(pred[m]), np.log10(life[m]))[0, 1]), 'in_1sigma': float(np.mean(np.abs(np.log10(pred[m]) - np.log10(life[m])) < sig[m]))}
    print(tag, {k: {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()} for k, v in out.items()}, flush=True); return out

def exam(net):
    cells, V = RD.load(); res = {}
    st = lambda rows, k: np.stack([r[k] for r in rows])
    rows = readings(cells, cycles=(100,)); S = np.array([[r['cycle'], r['charge_time']] for r in rows])
    P = RX.predict(net, st(rows, 'Q1'), st(rows, 'T1'), st(rows, 'Q0'), st(rows, 'T0'), S); res['cycle100'] = score(rows, P, 'cycle 100')
    rows_l = readings(cells, cycles=(), every=50); S = np.array([[r['cycle'], r['charge_time']] for r in rows_l])
    Pl = RX.predict(net, st(rows_l, 'Q1'), st(rows_l, 'T1'), st(rows_l, 'Q0'), st(rows_l, 'T0'), S)
    frac = np.array([r['cycle'] / r['life'] for r in rows_l]); life = np.array([r['life'] for r in rows_l]); pred = 10 ** Pl['log_life']; res['by_stage'] = {}
    for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.05)):
        m = (frac >= lo) & (frac < hi); e = pred[m] - life[m]
        res['by_stage'][f'{lo}-{hi}'] = {'n': int(m.sum()), 'rmse': float(np.sqrt(np.mean(e ** 2))), 'mpe': float(100 * np.mean(np.abs(e) / life[m])),
                                          'left_mae': float(np.mean(np.abs((pred[m] - np.array([r['cycle'] for r in rows_l])[m]) - (life[m] - np.array([r['cycle'] for r in rows_l])[m]))))}
    print('by stage of life', {k: {a: round(b, 1) if isinstance(b, float) else b for a, b in v.items()} for k, v in res['by_stage'].items()}, flush=True)
    res['severson_table2'] = SEVERSON; return res, rows, P, rows_l, Pl

if __name__ == '__main__':
    net = RX.load_net(sys.argv[1]); res = exam(net)[0]; json.dump(res, open(sys.argv[1].replace('.pt', '_life.json'), 'w'), indent=1)
