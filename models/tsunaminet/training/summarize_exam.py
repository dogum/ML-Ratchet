"""Exam summary tables (after the fact). Writes exam/exam_summary.json."""
import os, json, numpy as np
from _paths import ROOT
R = [r for r in json.load(open(os.path.join(ROOT, 'exam', 'exam_rows.json'))) if 'net' in r]
rng = np.random.default_rng(0)
def boot(vals_by_event, f, n=2000):
    ev = [v for v in vals_by_event if len(v)]
    est = f(np.concatenate(ev)); bs = []
    for _ in range(n):
        k = rng.integers(len(ev), size=len(ev)); bs.append(f(np.concatenate([ev[i] for i in k])))
    return [float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
out = dict(n_events=len(R), n_interface=sum(r['interface'] for r in R))
for sub, sel in [('all', lambda r: True), ('interface', lambda r: r['interface']), ('other', lambda r: not r['interface'])]:
    rr = [r for r in R if sel(r)]; o = dict(n=len(rr))
    for k in ('net', 'nnls', 'nnls_p1', 'dumb'):
        e = np.array([r[k]['mw'] - r['mw_true'] for r in rr])
        o[f'mw_mae_{k}'] = boot([[abs(x)] for x in e], np.mean); o[f'mw_bias_{k}'] = float(e.mean())
        c = [r[k].get('centroid_err_km') for r in rr if r[k].get('centroid_err_km') is not None]
        o[f'centroid_km_median_{k}'] = float(np.median(c)); o[f'centroid_km_mae_{k}'] = boot([[x] for x in c], np.mean)
        sc = [abs(r[k]['sc_err_km']) for r in rr if r[k].get('sc_err_km') is not None]
        o[f'sc_km_mae_{k}'] = boot([[x] for x in sc], np.mean)
    for q in ('68', '90'):
        o['mw_cov' + q] = float(np.mean([r['net']['mw' + q][0] <= r['mw_true'] <= r['net']['mw' + q][1] for r in rr]))
        cs = [r for r in rr if r['sc_true'] is not None]
        o['sc_cov' + q] = float(np.mean([r['net']['sc' + q][0] <= r['sc_true'] <= r['net']['sc' + q][1] for r in cs]))
    # amplitudes
    T = [[t for t in r['targets']] for r in rr]; nT = sum(len(t) for t in T); o['n_targets'] = nT; o['n_target_events'] = sum(1 for t in T if t)
    if nT:
        for k in ('net', 'nnls', 'nnls_p1', 'seismic', 'seismic_p1', 'dumb', 'ffm_twin'):
            L = [np.array([np.log10(max(t[k], 1e-5) / t['obs']) for t in tt]) for tt in T]
            o[f'amp_mae_{k}'] = boot([np.abs(l) for l in L], np.mean)
            o[f'amp_bias_{k}'] = float(np.mean(np.concatenate([l for l in L if len(l)])))
            o[f'amp_f2_{k}'] = boot([(np.abs(l) <= np.log10(2)).astype(float) for l in L], np.mean)
            # event-weighted (each event counts once)
            o[f'amp_mae_evw_{k}'] = float(np.mean([np.abs(l).mean() for l in L if len(l)]))
        o['amp_cov68_net'] = float(np.mean([t['net68'][0] <= t['obs'] <= t['net68'][1] for tt in T for t in tt]))
    out[sub] = o
json.dump(out, open(os.path.join(ROOT, 'exam', 'exam_summary.json'), 'w'), indent=1)
a = out['all']
print('Mw MAE  net %.3f [%.3f,%.3f] nnls %.3f nnls_p1 %.3f dumb %.3f | bias net %+.2f nnls %+.2f' % (*a['mw_mae_net'], a['mw_mae_nnls'][0], a['mw_mae_nnls_p1'][0], a['mw_mae_dumb'][0], a['mw_bias_net'], a['mw_bias_nnls']))
print('Mw cov68 %.2f cov90 %.2f | sc cov68 %.2f cov90 %.2f' % (a['mw_cov68'], a['mw_cov90'], a['sc_cov68'], a['sc_cov90']))
for k in ('net', 'nnls', 'nnls_p1', 'dumb'): print('centroid', k, 'median %.0f km' % a[f'centroid_km_median_{k}'], 'mae', np.round(a[f'centroid_km_mae_{k}'], 0), 'along-strike mae', np.round(a[f'sc_km_mae_{k}'], 0))
print('targets', a['n_targets'], 'events', a['n_target_events'])
for k in ('net', 'nnls', 'nnls_p1', 'seismic', 'seismic_p1', 'dumb', 'ffm_twin'):
    print('amp %-10s mae %.3f [%.3f,%.3f] evw %.3f bias %+.3f f2 %.2f' % (k, *a[f'amp_mae_{k}'], a[f'amp_mae_evw_{k}'], a[f'amp_bias_{k}'], a[f'amp_f2_{k}'][0]))
print('amp cov68 net %.2f' % a['amp_cov68_net'])
for sub in ('interface', 'other'):
    o = out[sub]; print(sub, o['n'], 'Mw mae net %.3f nnls %.3f nnls_p1 %.3f dumb %.3f' % (o['mw_mae_net'][0], o['mw_mae_nnls'][0], o['mw_mae_nnls_p1'][0], o['mw_mae_dumb'][0]), 'cov68 %.2f' % o['mw_cov68'],
          'amp net %.3f nnls %.3f seis %.3f seis_p1 %.3f dumb %.3f' % tuple(o.get(f'amp_mae_{k}', [np.nan])[0] for k in ('net', 'nnls', 'seismic', 'seismic_p1', 'dumb')) if o.get('n_targets') else '')
