"""THE EXAM. Run once, after the released rung and its dev calibration are frozen. Never chooses anything.
usage: python exam.py <released_rung>   (refuses to run twice: writes exam/exam_started.json first)
For every exam event: TsunamiNet posterior (Mw, centroid) and posterior-predictive amplitudes at held-out DARTs,
next to the SIFT-style NNLS inversion, seismic W-phase scaling and the dumb baseline; plus the USGS finite-fault twin
as context (not available in real time)."""
import os, sys, json, time, numpy as np
from _paths import DATA, ROOT
import tsim, realdata, evallib, baselines
rung = sys.argv[1]
OUT = os.path.join(ROOT, 'exam'); os.makedirs(OUT, exist_ok=True)
flag = os.path.join(OUT, 'exam_started.json')
if os.path.exists(flag) and '--resume' not in sys.argv:
    sys.exit('exam already started: ' + open(flag).read())
cal = json.load(open(os.path.join(ROOT, 'calibration.json')))
json.dump(dict(rung=rung, started=time.strftime('%Y-%m-%d %H:%M:%S'), calibration=cal), open(flag, 'w'))
geo = tsim.Geometry(); gf = tsim.GFBank(); net = evallib.load_net(rung)
EV = [e for e in json.load(open(os.path.join(DATA, 'events.json'))) if e['split'] == 'exam']
delay = cal['delay_frac']; gain = cal['amp_gain']; wf = cal['widen']; dumb = cal['dumb']
rows = []; SAVE = {}
for ev in EV:
    p = realdata.prepare(ev, gf, geo)
    ch = geo.chains[p['frame']['chain']]; j = p['frame']['j']
    cols_s = [((ch[j - tsim.HALF + m]['s'] - p['frame']['s0']) / 100.0 if 0 <= j - tsim.HALF + m < len(ch) else None) for m in range(tsim.NCOL)]
    SAVE[str(ev['id'])] = dict(X=p['X'], S=p['S'], M=p['M'], inputs=p['inputs'], input_ta=p['input_ta'], cols_s=cols_s)
    r = dict(id=ev['id'], name=ev['name'], interface=ev['interface'], mw_true=ev['mw'], sc_true=ev.get('ffm_s100'),
             ffm_lon=ev['ffm_lon'], ffm_lat=ev['ffm_lat'], inputs=p['inputs'], input_ta=p['input_ta'], n_targets=len(p['targets']))
    if not p['inputs']:
        r['skipped'] = 'no usable input DART within 290 min'; rows.append(r); print(ev['name'], 'skipped'); continue
    P = evallib.posterior(net, p['X'], p['S'], p['M'], 1000)[0]
    med = np.median(P, 0)
    def interval(k, q):
        lo, hi = np.percentile(P[:, k], [50 - q / 2, 50 + q / 2]); c = med[k]
        return float(c - wf[str(k)] * (c - lo)), float(c + wf[str(k)] * (hi - c))
    r['net'] = dict(mw=float(med[0]), mw68=interval(0, 68), mw90=interval(0, 90), sc=float(med[1]), sc68=interval(1, 68), sc90=interval(1, 90), fdeep=float(med[2]), logL=float(med[3]))
    lon, lat = geo.theta_to_latlon(p['frame'], med[1], np.clip(med[2], 0, 1)); r['net']['centroid'] = [float(lon), float(lat)]
    u, s, mw_n, slip = baselines.sift_nnls(gf, p); sc_n, fd_n = baselines.nnls_theta(slip)
    lo2, la2 = geo.theta_to_latlon(p['frame'], sc_n, fd_n)
    r['nnls'] = dict(mw=float(mw_n), sc=sc_n, centroid=[float(lo2), float(la2)])
    u1, s1, mw_n1, slip1 = baselines.sift_nnls(gf, p, delay_frac=delay, gain=gain); sc_n1, fd_n1 = baselines.nnls_theta(slip1)
    lo4, la4 = geo.theta_to_latlon(p['frame'], sc_n1, fd_n1)
    r['nnls_p1'] = dict(mw=float(mw_n1), sc=sc_n1, centroid=[float(lo4), float(la4)])
    lo3, la3 = geo.theta_to_latlon(p['frame'], 0.0, 0.5)
    r['seismic'] = dict(mw=ev['mw'], sc=0.0, centroid=[float(lo3), float(la3)])
    r['dumb'] = dict(mw=dumb['mw'], sc=0.0, centroid=[float(lo3), float(la3)])
    for k in ('net', 'nnls', 'nnls_p1', 'seismic', 'dumb'):
        if ev['ffm_lon'] is not None:
            r[k]['centroid_err_km'] = float(tsim.hav(ev['ffm_lat'], ev['ffm_lon'], r[k]['centroid'][1], r[k]['centroid'][0]))
            r[k]['sc_err_km'] = float(100 * (r[k]['sc'] - ev['ffm_s100']))
    # amplitude forecasts at held-out DARTs
    fc = evallib.amp_forecast(gf, geo, p, P, delay, gain=gain)
    us, ss = tsim.theta_slip(geo, p['frame'], baselines.seismic_theta(ev['mw']))
    fu, fs = realdata.ffm_units(ev, geo)
    T = []
    for t, f in zip(p['targets'], fc):
        d = t['ta'] * delay
        a_n = realdata.predict_amp(gf, t['site'], u, s, t['a'], t['b'], t['mask'])[0] if len(u) else 0.0          # classical: textbook GFs
        a_n1 = gain * realdata.predict_amp(gf, t['site'], u1, s1, t['a'], t['b'], t['mask'], d)[0] if len(u1) else 0.0
        a_s = realdata.predict_amp(gf, t['site'], us, ss, t['a'], t['b'], t['mask'])[0]
        a_s1 = gain * realdata.predict_amp(gf, t['site'], us, ss, t['a'], t['b'], t['mask'], d)[0]
        a_f = gain * realdata.predict_amp(gf, t['site'], fu, fs, t['a'], t['b'], t['mask'], d)[0]
        lg = np.log10(np.maximum(f[3], 1e-6)); m_, sd = np.median(lg), np.std(lg)
        sdw = np.sqrt((wf['amp'] * sd) ** 2 + cal['amp_model_sd'] ** 2)
        T.append(dict(site=t['site'], ta=t['ta'], obs=t['amp'], net=f[0], net68=[float(10 ** (m_ - sdw)), float(10 ** (m_ + sdw))],
                      nnls=a_n, nnls_p1=a_n1, seismic=a_s, seismic_p1=a_s1, dumb=dumb['amp'], ffm_twin=a_f))
    r['targets'] = T; rows.append(r)
    print(ev['name'][:30], 'Mw net %.2f nnls %.2f true %.2f' % (r['net']['mw'], mw_n, ev['mw']), 'targets', len(T), flush=True)
    json.dump(rows, open(os.path.join(OUT, 'exam_rows.json'), 'w'), indent=1, default=float)
json.dump(rows, open(os.path.join(OUT, 'exam_rows.json'), 'w'), indent=1, default=float)
import pickle; pickle.dump(SAVE, open(os.path.join(OUT, 'exam_inputs.pkl'), 'wb'))
print('exam done', len(rows))
