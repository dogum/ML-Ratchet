"""TransitNet exam on the rebuilt real set (15 Nov 2026), run once after the release choice: the v1.0 release (rung 3)
re-scored, rung 5 (least-squares polish) if it was kept, the least-squares transit fit (baseline.py) and the sqrt-depth
radius ratio, all on the same exam readings (SNR >= 7). Nothing here feeds any choice. Writes work/exam_v11.json."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, numpy as np
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.MODEL)
import ratchet_v11 as R, baseline as BL
D = _P.WORK + '/data/'

def summ(P, meta, snr, ls=None):
    k, b, aR = (np.array([m.get(x, np.nan) for m in meta], float) for x in ('k', 'b', 'aR'))
    E = dict(k=np.abs(10 ** P['log_k'] / k - 1), b=np.abs(P['b'] - b), aR=np.abs(10 ** P['log_aR'] / aR - 1))
    out = {}
    for lo, hi in ((7, 20), (20, 100), (100, 1e9), (7, 1e9)):
        m = (snr >= lo) & (snr < hi); key = f'snr{lo}-{int(hi) if hi < 1e9 else "inf"}'
        out[key] = {p: dict(n=int(np.isfinite(e[m]).sum()), median=float(np.nanmedian(e[m]))) for p, e in E.items()}
    return out

if __name__ == '__main__':
    S = json.load(open(D + 'split.json')); Mt = json.load(open(D + 'real_meta.json')); idx = [i for i in S['exam'] if Mt[i]['info']['snr'] >= 7]
    meta = [Mt[i] for i in idx]; kind = np.array([m['kind'] for m in meta]); isp = kind == 'planet'; isf = kind == 'FP'
    Zr = np.load(D + 'real.npz'); Z = {k: Zr[k][idx] for k in ('local', 'secondary', 'glob', 'oddeven', 's', 'cent', 'sc') if k in Zr.files}
    pm = [m for m in meta if m['kind'] == 'planet']; snr = np.array([m['info']['snr'] for m in pm]); pidx = np.where(isp)[0]
    log = json.load(open(_P.WORK + '/ratchet_v11.json')); rungs = ['r3'] + [r['rung'] for r in log if r['rung'] != 'r3' and r['accepted']]
    out = dict(n_planets=int(isp.sum()), n_fp=int(isf.sum()), n_fa=int((kind == 'FA').sum()))
    for rung in rungs:
        P = R.predict(Z); pp = P['p_planet']; m2 = isp | isf
        cls = dict(auc=R.auc(pp[m2], isp[m2]), planets_called_planet=float(np.mean(pp[isp] > 0.5)), fp_called_planet=float(np.mean(pp[isf] > 0.5)))
        Pp = {k: v[isp] for k, v in P.items()}
        if rung == 'r5': Pp = R.PO.polish_readings(Pp, np.asarray(Z['local'][pidx], float), np.asarray(Z['s'][pidx], float))[0]
        out[rung] = dict(cls, params=summ(Pp, pm, snr)); print(rung, json.dumps(out[rung]), flush=True)
    fits = BL.run([idx[i] for i in pidx])[0]
    L = {'log_k': np.log10([f['k'] for f in fits]), 'b': np.array([f['b'] for f in fits]), 'log_aR': np.log10([f['aR'] for f in fits])}
    out['least_squares'] = summ(L, pm, snr)
    depth = np.array([m['info']['depth'] for m in pm]); out['sqrt_depth'] = summ({'log_k': 0.5 * np.log10(np.clip(depth, 1e-9, None)), 'b': np.full(len(pm), np.nan), 'log_aR': np.full(len(pm), np.nan)}, pm, snr)
    print('least_squares', json.dumps(out['least_squares'])); json.dump(out, open(_P.WORK + '/exam_v11.json', 'w'), indent=1)
