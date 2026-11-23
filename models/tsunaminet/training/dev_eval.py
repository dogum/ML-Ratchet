"""Dev-event evaluation of a rung (real dev events only; never exam). Caches prepared dev events.
Outputs per event: Mw error vs USGS W-phase, along-strike centroid error vs the USGS finite-fault centroid (km),
68% coverage, and posterior-predictive amplitude errors at held-out dev DARTs. Also the SIFT-style NNLS baseline."""
import os, sys, json, pickle, numpy as np
from _paths import DATA, BANK
import tsim, realdata, evallib, baselines

def dev_events(gf, geo):
    fn = os.path.join(BANK, 'dev_prep.pkl')
    if os.path.exists(fn): return pickle.load(open(fn, 'rb'))
    EV = [e for e in json.load(open(os.path.join(DATA, 'events.json'))) if e['split'] == 'dev']
    out = []
    for ev in EV:
        p = realdata.prepare(ev, gf, geo)
        for t in p['targets']: t.pop('eta', None)
        p.pop('all'); out.append((ev, p))
    pickle.dump(out, open(fn, 'wb')); return out

def evaluate(net, gf, geo, delay_frac=0.0, amp_widen=1.0, n=400, forecasts=True, gain=1.0):
    rows = []
    for ev, p in dev_events(gf, geo):
        if not p['inputs']: continue
        P = evallib.posterior(net, p['X'], p['S'], p['M'], n)[0]
        med = np.median(P, 0); lo, hi = np.percentile(P, 16, 0), np.percentile(P, 84, 0)
        r = dict(id=ev['id'], name=ev['name'], interface=ev['interface'], n_in=len(p['inputs']), mw_true=ev['mw'], mw=float(med[0]),
                 mw_lo=float(lo[0]), mw_hi=float(hi[0]), sc_true=ev.get('ffm_s100'), sc=float(med[1]), sc_lo=float(lo[1]), sc_hi=float(hi[1]))
        r['mw_err'] = r['mw'] - r['mw_true']; r['sc_err_km'] = 100 * (r['sc'] - r['sc_true'])
        if forecasts and p['targets']:
            fc = evallib.amp_forecast(gf, geo, p, P, delay_frac, gain=gain)
            r['amp'] = [dict(site=t['site'], obs=t['amp'], med=f[0], lo=f[1], hi=f[2]) for t, f in zip(p['targets'], fc)]
        rows.append(r)
    E = lambda k: float(np.mean([abs(r[k]) for r in rows]))
    amp = [np.log10(a['med'] / a['obs']) for r in rows for a in r.get('amp', [])]
    summ = dict(n=len(rows), mw_mae=E('mw_err'), sc_mae_km=E('sc_err_km'), amp_mae_log10=float(np.mean(np.abs(amp))) if amp else None, n_amp=len(amp),
                mw_cov68=float(np.mean([r['mw_lo'] <= r['mw_true'] <= r['mw_hi'] for r in rows])))
    summ['dev_error'] = summ['mw_mae'] / 0.3 + summ['sc_mae_km'] / 100.0 + (summ['amp_mae_log10'] or 0) / 0.2
    return summ, rows

if __name__ == '__main__':
    rung = sys.argv[1]; geo = tsim.Geometry(); gf = tsim.GFBank(); net = evallib.load_net(rung)
    s, rows = evaluate(net, gf, geo)
    print(json.dumps(s, indent=1))
    for r in rows: print(r['name'][:28], 'Mw %.2f [%.2f,%.2f] true %.2f' % (r['mw'], r['mw_lo'], r['mw_hi'], r['mw_true']), 'sc %.0f km err' % r['sc_err_km'], 'n_amp', len(r.get('amp', [])))
