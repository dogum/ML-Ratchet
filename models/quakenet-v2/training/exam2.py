"""THE EXAM. Run once, at the end, after every choice is made. It chooses nothing.
QuakeNet v2 (released rung, dev-widened bars) against: QuakeNet v1.2 per station combined classically (dev-widened),
the travel-time locator (STA/LTA P + AIC S picks, ak135 grid search) and the dumb station centroid (dev-median depth 77 km
and Mw 5.85). Graded against USGS on QuakeNet v1.2's exam events: first 2 stations (every exam event with 2+), and first
3, 5 and all stations (the 400-event subset). Every rung's exam numbers are also reported, after the fact.
Writes exam_results.json and data/exam_preds.json. Refuses to run twice."""
import sys, os, json, time, numpy as np
from _paths import H; sys.path.insert(0, H + 'src')
import score2 as SC, train2 as T2, ratchet2 as RT, tt as TT

def strat(e, m):
    return {k: v[m] for k, v in e.items()}

def table(e, D, rows):
    out = {}
    for tag in ('2', '3', '5', 'all'):
        m = D['tag'][rows] == tag
        if not m.any(): continue
        s = SC.summary(strat(e, m)); z = 10 ** D['truth'][rows][m, 2]
        for nm, mm in (('shallow', z <= 70), ('deep', z > 70)):
            ee = strat(strat(e, m), mm); s[nm] = dict(n=int(mm.sum()), depth_med_km=float(np.nanmedian(ee['depth_km'])), depth_f2=float(np.nanmean(ee['logz'] < np.log10(2))), epi_med_km=float(np.median(ee['epi_km'])))
        out[tag] = s
    return out

if __name__ == '__main__':
    if os.path.exists(H + 'exam_results.json'): sys.exit('The exam has already been run (exam_results.json exists). It runs once.')
    t = time.time(); L = RT.log(); rel = [r for r in L if r['decision'] == 'released']; assert len(rel) == 1, 'mark exactly one released rung first'
    name = rel[0]['ckpt']; SF = json.load(open(H + 'data/sigma_factors.json')); assert SF['rung'] == name
    D = T2.load(H + 'data/exam_samples.npz'); rows = np.arange(len(D['K'])); res = dict(released=rel[0]['rung'], ckpt=name, n_samples=len(rows))
    P = SC.apply_factors(RT.preds(name, D, rows, real=True), SF['v2']); e2 = SC.errors(P, D, rows); res['quakenet_v2'] = table(e2, D, rows)
    V, C, Lc = SC.baselines(D, rows, real=True, tt_loc=True); fv = SF['v12_combined']
    for p in V: p['cov'] = p['cov'] * fv['epi'] ** 2; p['logz_sig'] *= fv['z']; p['mw_sig'] *= fv['mw']; p['t0_sig'] *= fv['t0']
    ev = SC.errors(V, D, rows); res['v12_combined'] = table(ev, D, rows); ec = SC.errors(C, D, rows); res['centroid'] = table(ec, D, rows); el = SC.errors(Lc, D, rows); res['tt_locator'] = table(el, D, rows)
    res['rungs'] = {}
    for r in L:
        if r['ckpt'] == 'anchor': continue
        e = SC.errors(RT.preds(r['ckpt'], D, rows, real=True), D, rows); res['rungs'][r['rung']] = table(e, D, rows)
    json.dump(res, open(H + 'exam_results.json', 'w'), indent=1)
    T = D['truth'][rows]; out = []
    for i, s in enumerate(rows):
        K = int(D['K'][s]); idx = D['idx'][s, :K]
        out.append(dict(id=str(D['event_id'][s]), tag=str(D['tag'][s]), K=K, truth=[float(v) for v in T[i]], v2=dict(lat=P[i]['lat'], lon=P[i]['lon'], depth=float(P[i]['depth']), mw=float(P[i]['mw']), t0=float(P[i]['t0']),
                    cov=np.asarray(P[i]['cov']).tolist(), logz_sig=float(P[i]['logz_sig']), mw_sig=float(P[i]['mw_sig']), t0_sig=float(P[i]['t0_sig'])),
                    v12=dict(lat=V[i]['lat'], lon=V[i]['lon'], depth=float(V[i]['depth']), mw=V[i]['mw'], t0=V[i]['t0']), tt=dict(lat=Lc[i]['lat'], lon=Lc[i]['lon'], depth=Lc[i]['depth'], t0=Lc[i]['t0']),
                    centroid=dict(lat=C[i]['lat'], lon=C[i]['lon']), nodes=idx.tolist()))
    json.dump(out, open(H + 'data/exam_preds.json', 'w')); print('exam done in %.0f s' % (time.time() - t)); print(json.dumps({k: res[k] for k in ('quakenet_v2', 'v12_combined', 'tt_locator', 'centroid')}, indent=1))
