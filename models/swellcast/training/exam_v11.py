"""The exam for v1.1, run once after the release choice (15 Nov 2026), with v1.0 re-scored on the same rebuilt windows.
numpy inference (swellcast.py): v1.0 = predict(prior=False), v1.1 = predict(). Baselines on the same windows: the dumb
dev-median guess (baseline.json's values) and the classical ridge fit (classical.py). Nothing here feeds any choice.
Writes work/exam_v11.json."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as _P
sys.path.insert(0, _P.MODEL)
import events as EV, swellcast as SC, classical as CL

def angdiff(a, b): return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)

def score(P, ev):
    D = np.array([e['D'] / 1e3 for e in ev]); t0 = np.array([e['t0_rel'] for e in ev]); dr = np.array([e['dir'] for e in ev]); hd = np.array([e['dir_coverage'] > 0.5 for e in ev])
    eD = np.log10(P['D_km']) - np.log10(D); et = P['t0'] - t0; ea = angdiff(P['dir'], dr)[hd]
    r = dict(n=len(ev), D_med_pct=float(np.nanmedian(np.abs(10 ** eD - 1)) * 100), logD_mae=float(np.nanmean(np.abs(eD))), t0_med_h=float(np.nanmedian(np.abs(et))), dir_med_deg=float(np.nanmedian(ea)))
    if 'logD_sig' in P: r.update(D_cov=float(np.mean(np.abs(eD) < P['logD_sig'])), t0_cov=float(np.mean(np.abs(et) < P['t0_sig'])))
    if 'kappa' in P: r['dir_cov'] = float(np.mean(angdiff(P['dir'], dr)[hd] < np.degrees(1 / np.sqrt(np.maximum(P['kappa'][hd], 1e-3)))))
    if 'detected' in P: r['detected'] = float(np.mean(P['detected']))
    return r

if __name__ == '__main__':
    net = SC.SwellCast(); dumb = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline.json')))['dumb_values']; out = {}
    for split, vis in (('exam', 'confirmed'), ('exam', True), ('exam', None), ('dev', 'confirmed')):
        ev = EV.load(split, vis); X = EV.windows(ev); key = f'{split}_{vis}'
        r = dict(v10=score(net.predict(X, prior=False), ev), v11=score(net.predict(X), ev),
                 dumb=score(dict(D_km=np.full(len(ev), 10 ** dumb['logD']), t0=np.full(len(ev), dumb['t0']), dir=np.full(len(ev), dumb['dir'])), ev))
        if vis == 'confirmed':
            C = [CL.read(x) for x in X]; r['classical'] = score(dict(D_km=np.array([c['D_km'] for c in C]), t0=np.array([c['t0'] for c in C]), dir=np.array([c['dir'] for c in C])), ev)
        out[key] = r; print(key, json.dumps({k: {a: round(b, 3) for a, b in v.items()} for k, v in r.items()}), flush=True)
    json.dump(out, open(os.path.join(_P.WORK, 'exam_v11.json'), 'w'), indent=1)
