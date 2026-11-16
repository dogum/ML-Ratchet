"""TransitNet ladder, 15 Nov 2026 rebuild: score the released rung 3 (numpy weights; the torch checkpoints were not kept)
and rung 5 (rung 3 + the least-squares polish, polish.py) on the frozen simulated set and the real dev set, with the
ladder's own rule: keep if the val2 objective rises by more than 0.01 AND dev log-loss is no more than 0.05 worse AND the
dev radius-ratio error no more than 1 point worse.

val2 was rebuilt from simbank.py with its own seed (99) and the v2 limb-darkening prior, on the rebuilt noise pool. The
noise pool is the same recipe (random.Random(7), 220 stars from each of 10 sectors) but MAST's current star lists and the
current planet/TOI catalogues exclude slightly different stars, so val2 is not bit-identical to the 3 Nov set; both rungs
are scored on the rebuilt one. Logs to work/ratchet_v11.json.
    python ratchet_v11.py r3          the release
    python ratchet_v11.py r5          the release + least-squares polish"""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, time, numpy as np
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.MODEL)
import transitnet as TNm, polish as PO
D = _P.WORK + '/data/'; B = _P.WORK + '/bank/'; VAL = B + 'val2.npz'; LOG = _P.WORK + '/ratchet_v11.json'
PARAMS = TNm.PARAMS

def auc(score, pos):
    s = np.asarray(score); pos = np.asarray(pos, bool); r = np.argsort(np.argsort(s)) + 1; n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))

def predict(Z, idx=None, polish=False, bs=256):
    net = TNm.TransitNet(); n = len(Z['s']); idx = np.arange(n) if idx is None else np.asarray(idx); R = {}
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]; V = {k: np.asarray(Z[k][j], np.float32) for k in ('local', 'secondary', 'glob', 'oddeven', 'cent') if k in Z}
        r = net.predict_views(V, np.nan_to_num(np.asarray(Z['s'][j], np.float32)), np.asarray(Z['sc'][j], np.float32) if 'sc' in Z else None)
        for k, v in r.items(): R.setdefault(k, []).append(v)
    R = {k: np.concatenate(v) for k, v in R.items()}
    if polish: R, _ = PO.polish_readings(R, np.asarray(Z['local'][idx], float), np.asarray(Z['s'][idx], float))
    return R

def val(polish):
    Z = dict(np.load(VAL)); cls, y, snr = Z['cls'], Z['y'], Z['snr']; ok = (cls == 0) & (snr >= 7)
    P = predict(Z, polish=False); out = {'auc': auc(P['p_planet'], cls == 0)}
    if polish:   # the polish only changes planet parameters: run it on the readings the parameter skill is scored on
        Pp = predict(Z, np.where(ok)[0], polish=True)
        for p in PARAMS: P[p] = P[p].copy(); P[p][ok] = Pp[p]
    for j, p in enumerate(PARAMS):
        e = np.abs(P[p][ok] - y[ok, j]); base = np.abs(y[ok, j] - np.median(y[ok, j])).mean()
        out[p] = {'mae': float(e.mean()), 'skill': float(1 - e.mean() / base)}
    hi = ok & (snr >= 100)
    for lo_, hi_ in ((7, 20), (20, 100), (100, 1e9)):
        m = ok & (snr >= lo_) & (snr < hi_)
        out[f'aR_frac_med_snr{lo_}'] = float(np.median(np.abs(10 ** (P['log_aR'][m] - y[m, 2]) - 1))); out[f'k_frac_med_snr{lo_}'] = float(np.median(np.abs(10 ** (P['log_k'][m] - y[m, 0]) - 1)))
        out[f'n_snr{lo_}'] = int(m.sum())
    out['objective'] = float(0.5 * (2 * out['auc'] - 1) + 0.5 * np.mean([max(0, out[p]['skill']) for p in PARAMS])); return out

def dev(polish):
    S = json.load(open(D + 'split.json')); Mt = json.load(open(D + 'real_meta.json')); idx = S['dev']; meta = [Mt[i] for i in idx]
    Zr = np.load(D + 'real.npz'); Z = {k: Zr[k][idx] for k in ('local', 'secondary', 'glob', 'oddeven', 's', 'cent', 'sc') if k in Zr.files}
    P = predict(Z, polish=polish); isp = np.array([m['kind'] == 'planet' for m in meta]); pp = np.clip(P['p_planet'], 1e-4, 1 - 1e-4)
    k, b, aR = (np.array([m.get(x, np.nan) for m in meta], float) for x in ('k', 'b', 'aR'))
    ek = np.abs(10 ** P['log_k'] / k - 1)[isp]; eb = np.abs(P['b'] - b)[isp]; ea = np.abs(10 ** P['log_aR'] / aR - 1)[isp]
    snr = np.array([m['info']['snr'] for m in meta])[isp]
    return {'n': len(idx), 'logloss': float(-np.mean(np.where(isp, np.log(pp), np.log(1 - pp)))), 'auc': auc(pp, isp),
            'k_frac_med': float(np.nanmedian(ek)), 'b_abs_med': float(np.nanmedian(eb)), 'aR_frac_med': float(np.nanmedian(ea)),
            'aR_frac_med_snr100': float(np.nanmedian(ea[snr >= 100])) if (snr >= 100).any() else None, 'n_snr100': int((snr >= 100).sum()),
            'fp_called_planet': float(np.mean(pp[~isp] > 0.5)), 'planets_called_planet': float(np.mean(pp[isp] > 0.5))}

if __name__ == '__main__':
    rung = sys.argv[1]; polish = rung == 'r5'; t = time.time()
    rec = dict(rung=rung, note={'r3': 'Released rung 3 (numpy weights), re-scored on the rebuilt val2 and dev',
                                'r5': 'Rung 3 + least-squares polish (polish.py): fit started at the network reading, limb darkening free, combined by covariance'}[rung],
               val=val(polish), dev=dev(polish))
    rec['objective'] = rec['val']['objective']; rec['eval_s'] = round(time.time() - t)
    log = json.load(open(LOG)) if os.path.exists(LOG) else []; inc = next((r for r in reversed(log) if r['accepted']), None)
    if inc is None: rec['accepted'] = True; rec['why'] = 'incumbent (released v1.0)'
    else:
        d, di = rec['dev'], inc['dev']; s_ok = rec['objective'] > inc['objective'] + 0.01; l_ok = d['logloss'] <= di['logloss'] + 0.05; k_ok = d['k_frac_med'] <= di['k_frac_med'] + 0.01
        rec['accepted'] = bool(s_ok and l_ok and k_ok); rec['vs'] = inc['rung']
        rec['why'] = (f"score {rec['objective']:.4f} vs {inc['objective']:.4f}" + ('' if s_ok else ' (needs +0.01)') + ('' if l_ok else f"; dev log-loss {d['logloss']:.3f} vs {di['logloss']:.3f}")
                      + ('' if k_ok else f"; dev radius-ratio error {100 * d['k_frac_med']:.1f}% vs {100 * di['k_frac_med']:.1f}%"))
    log.append(rec); json.dump(log, open(LOG, 'w'), indent=1); print(json.dumps(rec, indent=1))
